import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from finance_assistant.retrieval import search_chunks
from finance_assistant.generation import build_prompt, build_context,generate_answer

def main():
    evaluation_path = Path("data/evaluation/questions.json")
    embeddings_path = Path("data/processed/COCACOLA_2022_10K_embeddings.npy")
    chunks_path = Path("data/processed/COCACOLA_2022_10K_chunks.json")

    with evaluation_path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(embeddings_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    summaries = []

    for case in cases:
        print(f"\n{case['id']}")

        question_embedding = model.encode(
            case["question"], normalize_embeddings=True
        )

        results = search_chunks(chunks, embeddings, question_embedding, top_k=5)

        print(f"Question: {case['question']}")

        found_expected_page = False

        for rank, result in enumerate(results, start=1):
            page_match = (
                result["document"] == case["document"]
                and result["page_number"] in case["evidence_pages"]
            )

            if page_match:
                found_expected_page = True

            print(
                f"\nPosition {rank}, page {result['page_number']}, "
                f"Match with expected page: {page_match}"
            )
            print(result["text"])

        print(
            "\nExpected page found in top-5:",
            found_expected_page
        ) 

        context = build_context(results)
        prompt = build_prompt(case["question"], context)
        answer = generate_answer(prompt)

        value_match = answer.value == case["expected_value"]
        year_match = answer.year == case["expected_year"]

        units_match = answer.unit == case["expected_unit"]

        source_match = (
            answer.document == case["document"]
            and answer.page_number in case["evidence_pages"]
        )

        print("\nModel answer:", answer)
        print("Value matches:", value_match)
        print("Year matches:", year_match)
        print("Units matches:", units_match)
        print("Source matches:", source_match)

        summaries.append({
            "id": case["id"],
            "page_match": found_expected_page,
            "value_match": value_match,
            "year_match": year_match,
            "units_match": units_match,
            "source_match": source_match
        })
    print("\nSummary of results:")
    for summary in summaries:
        print(summary)    

if __name__ == "__main__":
    main()