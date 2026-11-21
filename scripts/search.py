import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks_path = Path("data/processed/COCACOLA_2022_10K_chunks.json")
    embeddings_path = Path("data/processed/COCACOLA_2022_10K_embeddings.npy")

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(embeddings_path)

    if len(chunks) != len(embeddings):
        print("Warning: The number of chunks and embeddings do not match.")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Number of embeddings: {len(embeddings)}")

    question = "What is the revenue of the company in 2022?"
    question_embedding = model.encode([question], normalize_embeddings=True)        
    
    scores = (embeddings @ question_embedding.T).flatten()
    top_indices = scores.flatten().argsort()[::-1][:5]
    
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]
    
        print(f"\nРезультат {rank}, сходство: {scores[index]:.3f}")
        print("Страница:", chunk["page_number"])
        print(chunk["text"])

if __name__ == "__main__":
    main()