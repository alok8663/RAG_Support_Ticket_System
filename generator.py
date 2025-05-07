import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(query, retrieved_texts):
    context = "\n".join(retrieved_texts)
    prompt = f"User Query: {query}\n\nRelevant Tickets:\n{context}\n\nGenerate a helpful support response."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']