import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_training_capability():
    print("--- GPU TRAINING TEST ---")
    
    # 1. Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not found. Check your venv!")
        return
    
    device = torch.device("cuda")
    print(f"Using Device: {torch.cuda.get_device_name(0)}")

    # 2. Load Model to VRAM
    print("Loading model to RTX 4060 VRAM...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 3. Simulate a single training batch
    # We create a fake sentence and try to pass it through the model
    dummy_input = "Hello, I am a therapist."
    inputs = tokenizer(dummy_input, return_tensors="pt").to(device)
    
    try:
        # Forward pass (the model 'thinks')
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Backward pass (the model 'learns')
        loss.backward()
        
        print("SUCCESS: Model processed text and calculated gradients on GPU!")
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"FAILED: Something went wrong: {e}")

if __name__ == "__main__":
    test_training_capability()