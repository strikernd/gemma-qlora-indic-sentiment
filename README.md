# Multilingual Indic Sentiment Fine-Tuning 

This repository contains the training and inference pipelines for fine-tuning the Gemma-3-1B-IT model on a multilingual sentiment classification task across 13 Indic languages. 

This project demonstrates how to perform Parameter-Efficient Fine-Tuning (PEFT) using QLoRA under strict VRAM constraints while maintaining high evaluation metrics.

## Performance Highlight
* F1-Macro Score: 0.923 
* Languages Supported: Assamese, Bodo, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu.

## Tech Stack
* Frameworks: PyTorch, Hugging Face Transformers, TRL (Transformer Reinforcement Learning)
* Optimization: PEFT (LoRA), bitsandbytes (4-bit Quantization), Gradient Accumulation
* Evaluation: Scikit-learn (Macro F1)

## Key Engineering Challenges Solved
1. Strict VRAM Constraints: Implemented 4-bit quantization, gradient checkpointing, and `batch_eval_metrics=True` to allow iterative evaluation across batches without Out-Of-Memory (OOM) crashes.
2. Instruction-Tuning Alignment: Designed a custom formatting pipeline using `apply_chat_template` to convert raw CSV data into a conversational prompt structure, maximizing the base model's instruction-following capabilities.
3. Robust Evaluation: Extracted and parsed generated raw-text outputs into standardized binary classifications to accurately compute F1 scores during validation.

## Repository Structure
* `train.py` - The main QLoRA fine-tuning script. Handles dataset formatting, 4-bit model initialization, LoRA adapter injection, and the SFT (Supervised Fine-Tuning) training loop.
* `infer.py` - The inference pipeline. Loads the base model, merges the saved LoRA adapter, and runs predictions on the test set.
* `.env.example` - Template for environment variables needed to access gated models.

## How to Run

### 1. Environment Setup
Install the required dependencies:
```bash
pip install torch pandas numpy python-dotenv datasets transformers peft trl bitsandbytes scikit-learn
