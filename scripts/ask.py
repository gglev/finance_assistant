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

    question = "What is the revenue of the company in 2022?"
    question_embedding = model.encode(question, normalize_embeddings=True)        
    
    results = search_chunks(chunks, embeddings, question_embedding, top_k=5)
    for rank, result in enumerate(results, start=1):
        print(f"\nРезультат {rank}, сходство: {result['score']:.3f}")
        print("Страница:", result["page_number"])
        print(result["text"])

    context = build_context(results)
    prompt = build_prompt(question, context)
    print("\nСформированный запрос:\n", prompt)

    answer = generate_answer(prompt)
    print("\nОтвет:\n", answer)

    if answer.value is None:
        print("В найденных фрагментах недостаточно информации для ответа на вопрос.")
    else:
        formatted_value = f"{answer.value:,}".replace(",", " ".replace(".", " "))
        print(f"Ответ: {answer.indicator} в {answer.year} году составляет {formatted_value} {answer.unit}")
        print(f"Источник: {answer.document}, страница {answer.page_number}")

if __name__ == "__main__":
    main()