import requests

API_URL = "https://ultra.dread.technology/v1/chat/completions"
API_KEY = "sk-damn-good-ultra-bread"  # Replace with your actual API key
MODEL = "claude-4.1-opus"  # Or any supported model
# MODEL = "bread-pg-1"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Start the conversation with a system prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Start chatting! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    messages.append({"role": "user", "content": user_input})
    payload = {
        "model": MODEL,
        "messages": messages
    }
    r = requests.post(API_URL, headers=headers, json=payload)
    if r.status_code == 200:
        reply = r.json()["choices"][0]["message"]["content"]
        print("\nAI:", reply)
        messages.append({"role": "assistant", "content": reply})
    else:
        print("Error:", r.status_code, r.text)
