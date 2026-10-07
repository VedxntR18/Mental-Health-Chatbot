import json
import os

# Configuration
LOGS_DIR = os.path.join("evaluation", "chat_logs")
OUTPUT_FILE = "data/processed/personal_training_data.txt"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def aggregate_logs():
    all_conversations = []
    
    # Check if folder exists and has files
    if not os.path.exists(LOGS_DIR) or not os.listdir(LOGS_DIR):
        print("No logs found to aggregate.")
        return

    for filename in os.listdir(LOGS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(LOGS_DIR, filename), 'r') as f:
                log_data = json.load(f)
                
                # Format each exchange for GPT-2 training
                for entry in log_data:
                    user_text = entry.get("user", "")
                    bot_text = entry.get("bot", "")
                    
                    # Formatting into a single string that GPT-2 can digest
                    # <s> and </s> help the model see start/end of thoughts
                    formatted_turn = f"<s>[USER]: {user_text} [BOT]: {bot_text}</s>\n"
                    all_conversations.append(formatted_turn)

    # Save to the processed data file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(all_conversations)
    
    print(f"Successfully aggregated {len(all_conversations)} turns into {OUTPUT_FILE}")

if __name__ == "__main__":
    aggregate_logs()