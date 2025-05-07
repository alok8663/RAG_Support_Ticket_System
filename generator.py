from openai import OpenAI
import os
from dotenv import load_dotenv

# Load your OpenAI API key from the .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(query, retrieved_tickets):
    context = "\n".join(retrieved_tickets)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant. Use the retrieved tickets to answer the user's query accurately."
        },
        {
            "role": "user",
            "content": f"My query is: {query}\n\nRelevant past tickets:\n{context}"
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message.content.strip()
