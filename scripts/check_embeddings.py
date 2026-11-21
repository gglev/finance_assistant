from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import numpy as np

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks_path = Path("data/processed/COCACOLA_2022_10K_chunks.json")

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings_path = Path("data/processed/COCACOLA_2022_10K_embeddings.npy")
    np.save(embeddings_path, embeddings)

    print("Embeddings saved to: ", embeddings_path)

    saved_embeddings = np.load(embeddings_path)

    print("Embeddings size: ", saved_embeddings.shape)
    print("Equality check: ", np.array_equal(embeddings, saved_embeddings))

if __name__ == "__main__":
    main()