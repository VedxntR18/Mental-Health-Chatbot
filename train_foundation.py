import torch
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

# --- 1. THE SCHEMA MAPPER (Standardizing 7 Different Formats) ---
def map_to_common_format(example):
    """Converts various dataset schemas into a single 'text' column."""
    # 1. Mental Health (Amod / marmikpandya)
    if "Context" in example and "Response" in example:
        return {"text": f"User: {example['Context']}\nAssistant: {example['Response']}"}
    # 2. Expert Mental Health (ShenLab)
    if "question" in example and "answer" in example:
        return {"text": f"User: {example['question']}\nAssistant: {example['answer']}"}
    # 3. General Chat (WizardLM / Dolly)
    if "instruction" in example and "output" in example:
        return {"text": f"User: {example['instruction']}\nAssistant: {example['output']}"}
    # 4. General Chat (LMSYS / ChatGPT Prompts)
    if "prompt" in example and "continuation" in example:
        return {"text": f"User: {example['prompt']}\nAssistant: {example['continuation']}"}
    # 5. Fallback for your Self-Correction log
    if "corrected" in example:
        return {"text": f"User: {example['user']}\nAssistant: {example['corrected']}"}
    
    return {"text": ""}

# --- 2. LOADING THE 7 STREAMS ---
def load_all_datasets():
    print("Inhaling 7 Knowledge Streams...")

    # MENTAL HEALTH TRIO
    mh1 = load_dataset("Amod/mental_health_counseling_conversations", split="train").select(range(500)).map(map_to_common_format)
    mh2 = load_dataset("ShenLab/MentalChat16K", split="train").select(range(500)).map(map_to_common_format)
    mh3 = load_dataset("marmikpandya/mental-health", split="train").select(range(500)).map(map_to_common_format)

    # GENERAL CHAT TRIO
    # Note: 'databricks-dolly-15k' is a high-quality open-source alternative for general Q&A
    gc1 = load_dataset("databricks/databricks-dolly-15k", split="train").select(range(500)).map(map_to_common_format)
    gc2 = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split="train").select(range(500)).map(map_to_common_format)
    gc3 = load_dataset("fka/awesome-chatgpt-prompts", split="train").select(range(200)).map(map_to_common_format)

    # PERSONALIZATION (The 7th Dataset: Your Self-Correction Log)
    pers_path = "data/processed/self_corrections.jsonl"
    if os.path.exists(pers_path):
        pers_ds = load_dataset("json", data_files=pers_path, split="train").map(map_to_common_format)
        # OVERSAMPLING: We multiply your corrections by 30 so the bot listens to YOU most of all
        pers_ds = concatenate_datasets([pers_ds] * 30)
        print(f"Personalization Active: {len(pers_ds)//30} unique corrections loaded.")
    else:
        pers_ds = None
        print("Personalization data not found. Skipping 7th stream.")

    # Merge everything
    streams = [mh1, mh2, mh3, gc1, gc2, gc3]
    if pers_ds: streams.append(pers_ds)
    
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
    output_dir="./mental_health_model",
    num_train_epochs=10,
    per_device_train_batch_size=2, # Safe for 8GB VRAM
    gradient_accumulation_steps=8,
    fp16=True,                    # RTX 40-series speed boost
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print(f"Starting Training on {len(data)} high-quality samples...")
trainer.train()

# Save the final consolidated brain
trainer.save_model("./mental_health_model")
tokenizer.save_pretrained("./mental_health_model")
print("SUCCESS: Foundation 7 Model Saved!")