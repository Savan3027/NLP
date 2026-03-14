from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

app = FastAPI()

classifier = joblib.load("models/ticket_classifier.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("models/faiss_index.bin")
train_texts = pd.read_csv("data/train_texts.csv")["text"].tolist()

class TicketRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_ticket(request: TicketRequest):

    embedding = embedding_model.encode([request.text]).astype("float32")

    prediction = classifier.predict(embedding)

    return {"predicted_label": prediction[0]}


@app.post("/similar")
def similar_tickets(request: TicketRequest):

    query_vector = embedding_model.encode([request.text]).astype("float32")

    if query_vector.shape[1] != index.d:
        return {"error": "Embedding dimension mismatch"}

    distances, indices = index.search(query_vector, 5)

    results = []

    for idx in indices[0]:
        if idx >= 0 and idx < len(train_texts):
            results.append(train_texts[idx])

    return {"similar_tickets": results}