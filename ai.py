import os
from google import genai
from ddgs import DDGS

# --- 1. CONFIGURATION ---
# IMPORTANT: Ensure your key is pasted inside the quotes without "YOUR_"
# Example: api_key="AIzaSy..." 
client = genai.Client(
    api_key="AIzaSyAXLd5mqGYHsQj2KQdwVK8JB9rN87L-6tI",
    http_options={'api_version': 'v1'}
)

def get_live_answer(user_query):
    # --- DIAGNOSTIC: Check what models you have access to ---
    # (You can remove this block once the script works)
    print("Checking available models...")
    for m in client.models.list():
        if "generateContent" in m.supported_actions:
            print(f"  > Available: {m.name}")
            # If you see a different model in the list, replace it below
    
    print(f"\nSearching internet for: {user_query}...")
    
    # --- 2. THE SEARCH (ddgs 2026 logic) ---
    search_snippets = []
    with DDGS() as ddgs:
        results = ddgs.text(user_query, max_results=3)
        for r in results:
            search_snippets.append(f"Source: {r['title']}\nContent: {r['body']}")
    
    context = "\n\n".join(search_snippets)

    # --- 3. THE PROMPT ---
    prompt = (
        f"You are a helpful assistant for a student at RAIT, Nerul. Use this data:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        f"Goal: Suggest specific local food spots near Nerul/Kamothe."
    )

    # --- 4. THE GENERATION (Updated to Gemini 2.5) ---
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Updated from 1.5
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Generation Error: {e}"

if __name__ == "__main__":
    print("--- Gemini Web Agent 2026 (v1 Stable) ---")
    query = input("You: ")
    if query.lower() != 'quit':
        print(f"\nAssistant: {get_live_answer(query)}")