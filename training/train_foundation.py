import torch
from pathlib import Path
import os
import json
import logging
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)

# --- PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "mental_health_model"
AUDIT_LOG = PROJECT_ROOT / "training_audit.log"
STATS_FILE = PROJECT_ROOT / "training_stats.csv"

# --- 1. SETUP LOGGING (The Audit Trail) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(AUDIT_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            if "loss" in logs:
                with open(STATS_FILE, "a") as f:
                    f.write(f"{state.global_step},{logs['loss']},{logs['learning_rate']}\n")

# --- 1. THE SCHEMA MAPPER (Standardizing 7 Different Formats) ---
def map_to_common_format(example):
    # 1. GC1: databricks-dolly-15k
    if "instruction" in example and "response" in example:
        user_part = example["instruction"]
        if example.get("context"): # Dolly often has extra context info
            user_part = f"{example['context']}\n{user_part}"
        return {"text": f"User: {user_part}\nAssistant: {example['response']}"}
    
    # 2. GC2: WizardLM (The most complex one)
    if "conversations" in example:
        # Extract human vs gpt turns from the list
        turns = example["conversations"]
        formatted_chat = ""
        for turn in turns:
            role = "User" if turn["from"] == "human" else "Assistant"
            formatted_chat += f"{role}: {turn['value']}\n"
        return {"text": formatted_chat.strip()}
    
    # 3. GC3: awesome-chatgpt-prompts
    if "act" in example and "prompt" in example:
        return {"text": f"User: Act as a {example['act']}.\nAssistant: {example['prompt']}"}
    
    return {"text": ""}

# --- 2. LOADING THE 7 STREAMS ---
def load_all_datasets():
    print("Inhaling 7 Knowledge Streams...")

    # # MENTAL HEALTH TRIO
    # mh1 = load_dataset("Amod/mental_health_counseling_conversations", split="train").select(range(500)).map(map_to_common_format)
    # mh2 = load_dataset("ShenLab/MentalChat16K", split="train").select(range(500)).map(map_to_common_format)
    # mh3 = load_dataset("marmikpandya/mental-health", split="train").select(range(500)).map(map_to_common_format)

    # GENERAL CHAT TRIO
    # Note: 'databricks-dolly-15k' is a high-quality open-source alternative for general Q&A
    gc1 = load_dataset("databricks/databricks-dolly-15k", split="train").select(range(500)).map(map_to_common_format)
    gc2 = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split="train").select(range(500)).map(map_to_common_format)
    gc3 = load_dataset("fka/awesome-chatgpt-prompts", split="train").select(range(200)).map(map_to_common_format)

    # PERSONALIZATION (The 7th Dataset: Your Self-Correction Log)
    # pers_path = "data/processed/self_corrections.jsonl"
    # if os.path.exists(pers_path):
    #     pers_ds = load_dataset("json", data_files=pers_path, split="train").map(map_to_common_format)
    #     # OVERSAMPLING: We multiply your corrections by 30 so the bot listens to YOU most of all
    #     pers_ds = concatenate_datasets([pers_ds] * 30)
    #     print(f"Personalization Active: {len(pers_ds)//30} unique corrections loaded.")
    # else:
    #     pers_ds = None
    #     print("Personalization data not found. Skipping 7th stream.")

    # Merge everything
    streams = [gc1, gc2, gc3]
    # if pers_ds: streams.append(pers_ds)
    
    final_ds = concatenate_datasets(streams).shuffle(seed=42)
    # Filter out any empty rows
    final_ds = final_ds.filter(lambda x: len(x['text']) > 5)
    return final_ds

# --- 3. TRAINING EXECUTION (RTX 4060 Optimized) ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load the base model to the GPU
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

data = load_all_datasets()
tokenized_ds = data.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=256),
    batched=True,
    remove_columns=data.column_names
)

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=15,
    per_device_train_batch_size=1, # Safe for 8GB VRAM
    gradient_accumulation_steps=8,
    fp16=True,                    # RTX 40-series speed boost
    learning_rate=8e-6,
    weight_decay=0.05,
    logging_steps=10,
    save_total_limit=2,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# --- 3. TRAINING PREP ---
data = load_all_datasets()

# 1. DO THE CHECK HERE (While 'text' still exists)
print("\n--- SAMPLE DATA CHECK ---")
for i in range(min(3, len(data))):
    print(f"Sample {i+1}:\n{data[i]['text'][:200]}...\n")
print("--------------------------\n")

# 2. NOW TOKENIZE (This deletes the 'text' column to save VRAM)
tokenized_ds = data.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=256),
    batched=True,
    remove_columns=data.column_names
)
print("--------------------------\n")

print(f"Starting Training on {len(data)} high-quality samples...")
trainer.train()

# Save the final consolidated brain
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print("SUCCESS: Foundation 7 Model Saved!")