from fastapi import FastAPI

app = FastAPI(title = "Finance assistant")

@app.get("/health")
def health():
    return {"status": "ok"}