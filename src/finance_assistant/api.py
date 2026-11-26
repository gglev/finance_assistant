from fastapi import FastAPI
import json
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from finance_assistant.generation import Answer, generate_answer, build_prompt, build_context
from finance_assistant.rag import answer_question

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@asynccontextmanager
async def lifespan(app: FastAPI):
    processed_direction = PROJECT_ROOT / "data" / "processed"

    chunks_path = processed_direction / "COCACOLA_2022_10K_chunks.json"
    embeddings_path = processed_direction / "COCACOLA_2022_10K_embeddings.npy"

    with chunks_path.open("r", encoding="utf-8") as file:
        app.state.chunks = json.load(file)

    app.state.embeddings = np.load(embeddings_path)
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")

    yield

app = FastAPI(title = "Finance assistant", lifespan=lifespan)

class Askrequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/ask",response_model = Answer)
def ask(payload: Askrequest) -> Answer:
    return answer_question(question=payload.question,
                            chunks=app.state.chunks,
                            embeddings=app.state.embeddings,
                            model = app.state.model 
                           )
