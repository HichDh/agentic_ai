from dataclasses import dataclass
from typing import Any, Dict

from .config import settings
from .lc_pipeline import load_vectorstore, build_rag_chain

@dataclass
class ToolResult:
    name: str
    payload: Dict[str, Any]

class Agent:
    def __init__(self, db_type="faiss"):
        self.vs = load_vectorstore(
            db_type=db_type,
            save_path=str(settings.index_path),
            model_name=settings.embedding_model
        )
        self.llm = None  # Optionally set a custom LLM

    def ask(self, question: str, k: int = 5) -> Dict[str, Any]:
        rag_chain = build_rag_chain(self.vs, llm=self.llm, k=k)
        answer = rag_chain.invoke(question)
        # Optionally, extract sources from retriever if needed
        return {"answer": answer.content if hasattr(answer, "content") else answer, "sources": []}