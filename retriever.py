from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

def load_tickets(path='tickets.json'):
    with open(path, 'r') as f:
        return json.load(f)

def build_index(tickets):
    texts = [f"{t['title']}. {t['issue']}. {t['resolution']}" for t in tickets]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, texts

def retrieve(query, index, texts, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k)
    return [texts[i] for i in I[0]]