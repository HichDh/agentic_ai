from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from .retriever import Retriever
from .agent import Agent

app = FastAPI(title="agentic_ai API")

class AskRequest(BaseModel):
    question: str
    k: int = 5

class IndexRequest(BaseModel):
    folder: str = "data/raw"

@app.post("/index")
def index(req: IndexRequest):
    Retriever().index_folder(req.folder)
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    res = Agent().ask(req.question, k=req.k)
    return res
