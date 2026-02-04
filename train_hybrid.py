import torch
import os
import logging
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)

# --- 1. SETUP LOGGING (The Audit Trail) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training_audit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            if "loss" in logs:
                with open("training_stats.csv", "a") as f:
                    f.write(f"{state.global_step},{logs['loss']},{logs['learning_rate']}\n")

# --- 2. LOAD & MERGE DATASETS ---
logger.info("Starting Data Ingestion...")
prof_ds = load_dataset("heliosbrahma/mental_health_chatbot_dataset", split="train")

personal_data_path = "data/processed/personal_training_data.txt"
if os.path.exists(personal_data_path):
    with open(personal_data_path, "r", encoding="utf-8") as f:
        personal_lines = [line.strip() for line in f.readlines() if line.strip()]
    personal_ds = Dataset.from_dict({"text": personal_lines})
    final_dataset = concatenate_datasets([prof_ds, personal_ds])
    logger.info(f"Merged {len(personal_lines)} personal lines.")
else:
    final_dataset = prof_ds
    logger.warning("Personal data not found.")

# --- 3. MODEL & TOKENIZER SETUP ---
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

logger.info("Loading GPT-2 to RTX 4060...")
model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")

# CRITICAL FIX: The tokenize function must explicitly handle padding/truncation
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

# CRITICAL FIX: remove_columns must be used to get rid of the 'text' string column
tokenized_ds = final_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=final_dataset.column_names
)

# --- 4. TRAINING ARGUMENTS ---
if not os.path.exists("training_stats.csv"):
    with open("training_stats.csv", "w") as f:
        f.write("step,loss,learning_rate\n")

training_args = TrainingArguments(
    output_dir="./mental_health_model",
    num_train_epochs=20,
    per_device_train_batch_size=2, # Safe for 8GB VRAM
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    learning_rate=3e-5,
    weight_decay=0.01,
    report_to="none"
)

# --- 5. INITIALIZE TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[PrinterCallback()]
)

# --- 6. START THE FINE-TUNING ---
logger.info("Starting Hybrid Fine-Tuning...")
trainer.train()

trainer.save_model("./mental_health_model")
tokenizer.save_pretrained("./mental_health_model")
logger.info("SUCCESS: Personalized Expert Model Saved to ./mental_health_model")