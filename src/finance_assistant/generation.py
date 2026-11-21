import httpx
from pydantic import BaseModel

class Answer(BaseModel):
    indicator: str
    year: int | None
    value: float | None
    unit: str
    document: str | None
    page_number: int | None

def build_context(results: list[dict]) -> str:
    fragments = []

    for number, result in enumerate(results, start=1):
        fragments.append(
            f"Фрагмент {number}\n"
            f"Документ: {result['document']}\n"
            f"Страница PDF: {result['page_number']}\n"
            f"{result['text']}"
        )

    return "\n\n".join(fragments)

def build_prompt(question: str, context: str) -> str:
    return (
        f"Вопрос: {question}\n\n"
        f"Контекст:\n{context}\n\n"
        "Верни ответ по заданной JSON-схеме.\n"
        "В value укажи числовое значение без разделителей тысяч.\n"
        "Сохраняй масштаб единиц источника и указывай его в unit.\n"
        "Если ответа нет в контексте, value должно быть null.\n"
    )

def generate_answer(prompt: str) -> Answer:
    response = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:8b",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "think": False,
            "format": Answer.model_json_schema()
        },
        timeout=120.0,
    )

    response.raise_for_status()

    data = response.json()

    return Answer.model_validate_json(data["message"]["content"])