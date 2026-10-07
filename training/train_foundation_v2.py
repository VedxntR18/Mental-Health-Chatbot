import torch
from pathlib import Path
import os
import json
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

# --- PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "mental_health_model"

# --- 1. THE SCHEMA MAPPER ---
def map_to_common_format(example):
    if "Context" in example and "Response" in example:
        return {"text": f"User: {example['Context']}\nAssistant: {example['Response']}"}
    if "question" in example and "answer" in example:
        return {"text": f"User: {example['question']}\nAssistant: {example['answer']}"}
    if "instruction" in example and "output" in example:
        return {"text": f"User: {example['instruction']}\nAssistant: {example['output']}"}
    if "prompt" in example and "continuation" in example:
        return {"text": f"User: {example['prompt']}\nAssistant: {example['continuation']}"}
    return {"text": ""}

# --- 2. LOADING 7 STREAMS (Full Power) ---
def load_v2_data():
    print("Inhaling Foundation V2.1 Knowledge Streams...")
    
    # 1600-2000 rows per expert stream
    mh1 = load_dataset("Amod/mental_health_counseling_conversations", split="train").select(range(1600)).map(map_to_common_format)
    mh2 = load_dataset("ShenLab/MentalChat16K", split="train").select(range(1600)).map(map_to_common_format)
    mh3 = load_dataset("marmikpandya/mental-health", split="train").select(range(1600)).map(map_to_common_format)
    gc1 = load_dataset("databricks/databricks-dolly-15k", split="train").select(range(2000)).map(map_to_common_format)
    gc2 = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split="train").select(range(2000)).map(map_to_common_format)
    gc3 = load_dataset("fka/awesome-chatgpt-prompts", split="train").select(range(400)).map(map_to_common_format)

    # PERSONALIZATION (100x Oversampling)
    pers_path = PROJECT_ROOT / "evaluation" / "self_corrections.jsonl"
    if os.path.exists(pers_path):
        pers_ds = load_dataset("json", data_files=pers_path, split="train")
        # FIXED: Added the missing closing parenthesis here
        pers_ds = concatenate_datasets([pers_ds] * 100) 
        print(f"Personalization Weighted: {len(pers_ds)} records injected.")
    else:
        print("WARNING: No self_corrections.jsonl found!")
        pers_ds = None

    streams = [mh1, mh2, mh3, gc1, gc2, gc3]
    if pers_ds: streams.append(pers_ds)
    
    return concatenate_datasets(streams).shuffle(seed=42)

# --- 3. TRAINING CONFIG (RTX 4060 Beast Mode) ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

data = load_v2_data()

# TOKENIZATION
tokenized_ds = data.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=1024),
    batched=True,
    remove_columns=data.column_names
)

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=30,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    fp16=True,
    learning_rate=2e-5,
    weight_decay=0.02,
    logging_steps=5,
    save_total_limit=1,
    # ADDED: Gradient Checkpointing to save VRAM for 1024 context length
    gradient_checkpointing=True,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print(f"Starting V2.1 Heavy Training on {len(data)} samples...")
trainer.train()
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print("SUCCESS: Foundation V2.1 Model Ready!")