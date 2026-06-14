import os
import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import f1_score

# ---------------------------------------------------------
# 1. Environment & GPU Configuration

# Optimizes memory allocation for PyTorch to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Uncomment this if you have multiple GPUs and are facing DDP (DistributedDataParallel) crashes during LoRA training.
# This forces HuggingFace to only see 1 GPU, bypassing the parallelization errors.
# torch.cuda.device_count = lambda: 1

# Load HuggingFace Token from local .env file instead of Kaggle Secrets
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found. Make sure you have a .env file with HF_TOKEN=your_token")

# ---------------------------------------------------------
# 2. Dataset Loading & Preparation

# Dummy paths - replace these with your actual local paths
train_data_path = "./data/train.csv"

print(f"Loading data from {train_data_path}...")
train_dataset_full = load_dataset('csv', data_files=train_data_path, split="train")

# Splitting 10% of the training data for evaluation
split = train_dataset_full.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
eval_data = split["test"]

# ---------------------------------------------------------
# 3. Tokenizer & Formatting

model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512

# Language mapping to convert codes to full language names for better LLM comprehension
lang_dict = {
    "as": "Assamese", "bd": "Bodo", "bn": "Bengali", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu"
}

def train_formatter(sample):
    """
    Formats the raw dataset rows into the instruction-tuned chat template.
    """
    chat = [
      {
          "role": "user", 
          "content": f"You are an expert in emotional analysis of Indic Languages. Give special attention to tricky texts, where one part negates the other. Find the sentiment of the following text in {lang_dict[sample['language']]} into either 'Positive' or 'Negative'. Text: \"{sample['sentence']}\" Sentiment(Positive/Negative)?"
      },
      {
          "role": "assistant", 
          "content": f"{sample['label']}" 
      }
    ]
    sample["formatted"] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return sample

print("Formatting chat templates...")
new_train = train_data.map(train_formatter, remove_columns=["sentence", "label", "language"])
new_eval = eval_data.map(train_formatter, remove_columns=["sentence", "label", "language"])

# ---------------------------------------------------------
# 4. Model Initialization & PEFT (LoRA)

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.2,
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj','v_proj','k_proj','o_proj','gate_proj','up_proj','down_proj']
)

# Wrap the base model with the LoRA adapter
model = get_peft_model(model, lora_config)

# ---------------------------------------------------------
# 5. Metrics Evaluation

# Global variables are retained here because batch_eval_metrics=True 
# evaluates metrics iteratively across batches to save VRAM.
texts = []
all_preds = []
all_labels = []
counts = {"Positive": 0, "Negative": 0, "Unknown": 0}

def extract_label(text):
    global texts
    text = text.strip().lower()
    texts.append(text)
    if "model\npositive" in text:
        counts["Positive"] += 1
        return "Positive"
    elif "model\nnegative" in text:
        counts["Negative"] += 1
        return "Negative"
    else:
        counts["Unknown"] += 1
        return "Unknown"

def compute_metrics(eval_pred, compute_result):
    global all_preds, all_labels
    logits, labels = eval_pred
  
    # Get predicted token ids from logits
    predicted_ids = np.argmax(logits.detach().cpu(), axis=-1)
    
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    for t in decoded_preds:
        all_preds.append(extract_label(t))
    for t in decoded_labels:
        all_labels.append(extract_label(t))
    
    # compute_result is True only on the final batch of the epoch
    if compute_result:  
        f1 = f1_score(all_labels, all_preds, average="macro")
        all_preds = []   # reset for next epoch
        all_labels = []
        return {"f1_macro": f1}

# ---------------------------------------------------------
# 6. SFT Trainer Setup & Execution

sft_config = SFTConfig(
    output_dir="nppe_model_checkpoints",
    max_length=512,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    dataset_text_field="formatted",
    ddp_find_unused_parameters=False,
    packing=False, 
    num_train_epochs=10,
    bf16=True,
    dataloader_num_workers=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    gradient_checkpointing=True,
    batch_eval_metrics=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    seed=42,
    completion_only_loss=True, 
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=new_train,
    args=sft_config,
    processing_class=tokenizer,
    eval_dataset=new_eval,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# ---------------------------------------------------------
# 7. Save Final Model

final_model_path = "./saved_model_gemma_finetuned"
print(f"Training complete. Saving final model to {final_model_path}...")

# Saves the LoRA weights
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print("Model saved successfully!")
