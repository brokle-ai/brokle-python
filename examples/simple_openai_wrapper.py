import os
from openai import OpenAI
from brokle import wrap_openai

os.environ["BROKLE_API_KEY"] = "bk_test"
os.environ["BROKLE_HOST"] = "http://localhost:8080"

os.environ["OPENAI_API_KEY"] = "sk-proj-testkeyforlocaldebuggingonly"

# Create a client (uses your OPENAI_API_KEY environment variable)
client = OpenAI()
client = wrap_openai(client)

# Chat completion example
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about the ocean."},
    ],
)

print(response.choices[0].message.content)
