from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from .retriever import Retriever
from .llm import ProviderRouter
from .config import settings

@dataclass
class ToolResult:
    name: str
    payload: Dict[str, Any]

class Agent:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = ProviderRouter()

    def _tool_retrieve(self, question: str, k: int) -> ToolResult:
        hits = self.retriever.retrieve(question, k=k)
        sources = [{"score": s, "text": d.text[:400], "meta": d.meta} for d,s in hits]
        context = "\n\n".join([f"[Source {i}] {s['text']}" for i,s in enumerate(sources, 1)])
        return ToolResult(name="retrieve", payload={"context": context, "sources": sources})

    def ask(self, question: str, k: int = 5) -> Dict[str, Any]:
        # Plan → Retrieve → Synthesize (simple, explicit chain for readability)
        retrieve_res = self._tool_retrieve(question, k)
        system = settings.system_prompt
        user = f"Question: {question}\n\nContext:\n{retrieve_res.payload['context']}"
        answer = self.llm.generate(system=system, user=user)
        return {"answer": answer, "sources": retrieve_res.payload["sources"]}
