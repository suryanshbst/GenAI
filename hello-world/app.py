import os
import time
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Groq API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

# Configure the client to route to Groq's OpenAI-compatible gateway
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Initialize conversation history with system behavior
messages = [
    {"role": "system", "content": "You are a helpful assistant talking to Suryansh."}
]

print("Groq Chat initialized successfully! Type 'exit' to quit.\n")

# Main conversation loop
while True:
    user_input = input("Suryansh: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
        
    messages.append({"role": "user", "content": user_input})
    
    try:
        # Request inference from Llama-3.3 hosted on Groq's hardware
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        
        reply = response.choices[0].message.content
        print(f"\nAI: {reply}\n")
        
        messages.append({"role": "assistant", "content": reply})
        
    except RateLimitError:
        print("\n[System Warning: Hit Groq rate limits. Waiting 5s before retrying...]\n")
        messages.pop() 
        time.sleep(5)