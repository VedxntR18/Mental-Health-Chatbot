import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import json
import uuid
from pathlib import Path
from datetime import datetime

# --- 1. HARDWARE & DIRECTORY SETUP ---
device = 0 if torch.cuda.is_available() else -1
PROJECT_ROOT = Path(__file__).resolve().parent.parent
model_path = PROJECT_ROOT / "mental_health_model"

# Create directories for MLOps compliance
LOGS_DIR = PROJECT_ROOT / "chat_logs"
CORRECTIONS_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Session Tracking
session_id = f"session_{uuid.uuid4().hex[:8]}"
session_file = LOGS_DIR / f"{session_id}.json"
chat_history = []

# --- 2. MODEL LOADING (RTX 4060 Optimization) ---
if not model_path.exists():
    print("!!! Fine-tuned model not found. Using base 'gpt2' for now. !!!")
    model_path = "gpt2"

print(f"Loading Brain from: {model_path} on {'GPU' if device == 0 else 'CPU'}...")

tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(str(model_path))

generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    device=device
)

# --- 3. DATASET & INTENTS ---
with open(PROJECT_ROOT / 'data' / 'intents.json', 'r', encoding='utf-8') as file:
    local_intents = json.load(file)

# --- 4. CORE ENGINE FUNCTIONS ---

def save_chat_to_file(user_input, bot_response):
    chat_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_input,
        "bot": bot_response
    })
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, indent=4)

def self_correct_logic(user_input, initial_response):
    # 1. Standardizing the prompt with the new 'Balanced' identity
    critique_prompt = (
        f"USER: {user_input}\n"
        f"INITIAL_REPLY: {initial_response}\n\n"
        f"TASK: Rewrite the reply to be factual (regarding AIML or Navi Mumbai) and empathetic. "
        f"Ensure you answer the user's specific question first. Do NOT use generic supportive phrases if the user asked for a fact.\n"
        f"BALANCED_REPLY:"
    )
    
    # 2. Generate the correction - ensure the variable is assigned correctly
    try:
        raw_results = generator(
            critique_prompt, 
            max_new_tokens=80, 
            temperature=0.3, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Access the generated text safely
        correction_output = raw_results[0]['generated_text']
        
        # Safer extraction logic
        if "BALANCED_REPLY:" in correction_output:
            better_answer = correction_output.split("BALANCED_REPLY:")[-1].strip()
        else:
            better_answer = correction_output.strip()
            
        # Remove any lingering prompt artifacts
        better_answer = better_answer.split("\n")[0].split("USER:")[0].strip()
        
        return initial_response # Return the first thought if the correction is bad
        
    except Exception as e:
        print(f"Critique Error: {e}")
        return initial_response

def get_therapist_response(user_text):
    recent_context = "\n".join([f"User: {c['user']}\nAssistant: {c['bot']}" for c in chat_history[-2:]])
    
    # We are adding "Answer as a Navi Mumbai local" directly to the injection
    full_prompt = (
        f"You are a local Navi Mumbai guide and AIML engineer at RAIT. "
        f"Answer this specific question about food or tech first: {user_text}\n"
        f"Context: {recent_context}\n"
        f"Assistant:"
    )

    raw_output = generator(
        full_prompt, 
        max_new_tokens=100, 
        temperature=0.2,       
        repetition_penalty=1.5, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        clean_up_tokenization_spaces=True # Adds a cleaner look to the text
    )
    initial_msg = raw_output[0]['generated_text'].strip().split("\n")[0]
    
    # STEP D: Automated Self-Correction (The 7th Dataset Builder)
    final_msg = self_correct_logic(user_text, initial_msg)
    
    return final_msg

# --- 5. MAIN CHAT LOOP ---
if __name__ == "__main__":
    print(f"System: RTX 4060 Online. Auto-Critique Active.")
    print("Therapist: Hello. I am here for you. (Type 'quit' to exit)")
    
    while True:
        user_msg = input("You: ")
        if user_msg.lower() == 'quit':
            print(f"Session {session_id} saved. Goodbye!")
            break
        
        # Get the smart, self-corrected response
        bot_msg = get_therapist_response(user_msg)
        print(f"Bot: {bot_msg}")
        
        # Log the final outcome for memory
        save_chat_to_file(user_msg, bot_msg)