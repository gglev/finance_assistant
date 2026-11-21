import numpy as np

def search_chunks(
        chunks: list[dict],
        embeddings: np.ndarray,
        question_embedding: np.ndarray,
        top_k: int = 5
) -> list[dict]:
    if len(chunks) != len(embeddings):
        raise ValueError("The number of chunks and embeddings do not match.")

    if not 1 <= top_k <= len(chunks):
        raise ValueError(f"top_k must be between 1 and {len(chunks)}")
    
    scores = (embeddings @ question_embedding).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for index in top_indices:
        chunk = chunks[index]
        results.append({
            "document": chunk["document"],
            "page_number": chunk["page_number"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"],
            "score": scores[index]
        })

    return results