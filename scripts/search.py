import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from finance_assistant.retrieval import search_chunks

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks_path = Path("data/processed/COCACOLA_2022_10K_chunks.json")
    embeddings_path = Path("data/processed/COCACOLA_2022_10K_embeddings.npy")

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(embeddings_path)

    question = "What is the revenue of the company in 2022?"
    question_embedding = model.encode(question, normalize_embeddings=True)        
    
    results = search_chunks(chunks, embeddings, question_embedding, top_k=5)
    for rank, result in enumerate(results, start=1):
        print(f"\nРезультат {rank}, сходство: {result['score']:.3f}")
        print("Страница:", result["page_number"])
        print(result["text"])

if __name__ == "__main__":
    main()