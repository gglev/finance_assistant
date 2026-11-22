import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from finance_assistant.retrieval import search_chunks
from finance_assistant.generation import build_prompt, build_context
from finance_assistant.generation import (build_prompt, build_context, generate_answer)

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks_path = Path("data/processed/COCACOLA_2022_10K_chunks.json")
    embeddings_path = Path("data/processed/COCACOLA_2022_10K_embeddings.npy")

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(embeddings_path)

    question = "What were Coca-Cola's net operating revenues in 2025?"
    question_embedding = model.encode(question, normalize_embeddings=True)        
    
    results = search_chunks(chunks, embeddings, question_embedding, top_k=5)
    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}, similarity: {result['score']:.3f}")
        print("Page:", result["page_number"])
        print(result["text"])

    context = build_context(results)
    prompt = build_prompt(question, context)
    print("\Prompt:\n", prompt)

    answer = generate_answer(prompt)
    print("\nAnswer:\n", answer)

    if answer.value is None:
        print("In the found fragments, there is not enough information to answer the question.")
    else:
        formatted_value = (
            f"{answer.value:,}"
            .replace(",", " ")
            .replace(".", ",")
        )
        print(f"Answer: {answer.indicator} in {answer.year} year is {formatted_value} {answer.unit}")
        print(f"Source: {answer.document}, page {answer.page_number}")

if __name__ == "__main__":
    main()