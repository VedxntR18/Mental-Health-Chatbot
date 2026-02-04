import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

# 1. Load the Professional Dataset
print("Loading mental health dataset...")
dataset = load_dataset("heliosbrahma/mental_health_chatbot_dataset")

# 2. Setup Tokenizer & Model for your RTX 4060
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model directly to GPU
model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")

# 3. Preprocess for Causal Language Modeling
def tokenize_function(examples):
    # This prepares the text so the model learns to predict the next word in a therapy context
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Training Arguments (Optimized for 8GB VRAM)
training_args = TrainingArguments(
    output_dir="./mental_health_model",
    overwrite_output_dir=True,
    num_train_epochs=3,           # How many times to study the dataset
    per_device_train_batch_size=4, # Safe for 8GB VRAM
    gradient_accumulation_steps=4, # Effectively makes batch size 16 without crashing memory
    learning_rate=5e-5,
    fp16=True,                    # Uses RTX 40-series tensor cores for 2x speed
    logging_steps=10,
    save_steps=100,
    push_to_hub=False,
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 6. TRAIN!
print("Starting Fine-Tuning on RTX 4060...")
trainer.train()

# 7. Save the specialized 'Expert' weights
trainer.save_model("./mental_health_model")
tokenizer.save_pretrained("./mental_health_model")
print("Expert Brain Saved to ./mental_health_model")