import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from src.query_engine.schemas import AssistantRequest, AssistantResponse
from src.rag.assistant import RAGAssistant

app = FastAPI(title="Excel RAG Analytics Assistant", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AssistantResponse)
def ask(request: AssistantRequest):
    try:
        result = RAGAssistant().ask(request.question)
        return AssistantResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
