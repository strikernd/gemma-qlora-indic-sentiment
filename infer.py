import os
import torch
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from huggingface_hub import login
from peft import PeftModel

# ---------------------------------------------------------
# 1. Environment & GPU Configuration

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# torch.cuda.device_count = lambda: 1

# Base Gemma model requires HF Token to download
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found. Required if base model is gated.")

# ---------------------------------------------------------
# 2. Paths & Dataset Loading

# Replace with your actual paths
test_data_path = "./data/test.csv"
model_dir = "./saved_model_gemma_finetuned"
output_csv = "./submission.csv"

print(f"Loading test data from {test_data_path}...")
test_data = load_dataset('csv', data_files=test_data_path, split="train")

# ---------------------------------------------------------
# 3. Model & Tokenizer Initialization

model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16
)

print(f"Loading LoRA weights from {model_dir}...")
model = PeftModel.from_pretrained(base_model, model_dir)
model.eval()

# ---------------------------------------------------------
# 4. Prediction Pipeline

lang_dict = {
    "as": "Assamese", "bd": "Bodo", "bn": "Bengali", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu"
}

def predict(sample):
    chat = [
        {
            "role": "user", 
            "content": f"You are an expert in emotional analysis of Indic Languages. Give special attention to tricky texts, where one part negates the other. Find the sentiment of the following text in {lang_dict[sample['language']]} into either 'Positive' or 'Negative'. Text: \"{sample['sentence']}\" Sentiment(Positive/Negative)?"
        }
    ]
    
    # Added return_dict=True so inputs["input_ids"] works smoothly
    inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=1,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return prediction

# ---------------------------------------------------------
# 5. Execution & Saving

predictions = []
ids = []

print("Running inference...")
for sample in test_data:
    pred = predict(sample)
    ids.append(sample["ID"])
    
    if pred == "Positive":
        predictions.append(1)
    else:
        predictions.append(0)

print(f"Sample outputs: {predictions[:5]}") 

# Save submission
final = pd.DataFrame({'ID': ids, 'label': predictions})
final.to_csv(output_csv, index=False)
print(f"Predictions successfully saved to {output_csv}")
