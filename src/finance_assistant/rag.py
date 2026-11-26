import numpy as np
from sentence_transformers import SentenceTransformer
from finance_assistant.retrieval import search_chunks
from finance_assistant.generation import Answer,build_context, build_prompt, generate_answer

def answer_question(
        question:str,
        chunks: list[dict],
        embeddings: np.ndarray,
        model: SentenceTransformer) -> Answer:
    question_embedding = model.encode(question, normalize_embeddings= True)

    results = search_chunks(chunks,embeddings,question_embedding, top_k= 5)

    context = build_context(results)
    prompt = build_prompt(question, context)

    return generate_answer(prompt)