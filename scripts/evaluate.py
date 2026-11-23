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

    retrieval_scores = []

    for case in cases:
        print(f"\n{case['id']}")

        question_embedding = model.encode(
            case["question"], normalize_embeddings=True
        )

        results = search_chunks(chunks, embeddings, question_embedding, top_k=5)

        if case.get("relevant_chunks"):
            first_relevant_rank = None

            for rank,result in enumerate(results, start=1):
                chunk_match = (
                    result["document"] == case["document"]
                    and
                    {
                        "page_number": result["page_number"],
                        "chunk_number": result["chunk_number"]
                    } in case["relevant_chunks"]
                )

                if chunk_match:
                    first_relevant_rank = rank
                    break

            hit_at_5 = first_relevant_rank is not None
            reciprocal_rank = (
                1 / first_relevant_rank
                if first_relevant_rank is not None
                else 0.0
            )

            retrieval_scores.append({
                "id": case["id"],
                "hit_at_5": hit_at_5,
                "reciprocal_rank": reciprocal_rank,
            })

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

        if case["expected_value"] is None:
            refusal_match = answer.value is None

            print("\nModel answer:", answer)
            print("Correct refusal:", refusal_match)

            summaries.append({
                "id": case["id"],
                "refusal_match": refusal_match})

            continue

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
    # if retrieval_scores:
    #     mrr = sum(score["reciprocal_rank"] for score in retrieval_scores) / len(retrieval_scores)
    #     print("MRR@5", mrr)

    print("\nSummary of results:")
    for summary in summaries:
        print(summary)    

    if retrieval_scores:
        print("\nRetrieval results:")
        print(f"{'question':<32}{'Hit@5':>8}{'RR@5':>8}")

        for score in retrieval_scores:
            print(
                f"{score['id']:<32}"
                f"{str(score['hit_at_5']):>8}"
                f"{score['reciprocal_rank']:>8.3f}"
            )

        hit_rate = sum(score["hit_at_5"] for score in retrieval_scores) / len(retrieval_scores)

        mrr = sum(score["reciprocal_rank"] for score in retrieval_scores) / len(retrieval_scores)

        print(f"Labeled questions: {len(retrieval_scores)}")
        print(f"Hit@5: {hit_rate:.3f}")
        print(f"MRR@5: {mrr:.3f}")

    report_path = Path("reports/evaluation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)

    print("Report saved to: ", report_path)

if __name__ == "__main__":
    main()