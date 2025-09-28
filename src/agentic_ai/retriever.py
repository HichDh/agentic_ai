from .config import settings
from .lc_pipeline import load_vectorstore


class Retriever:
    def __init__(self, db_type="faiss"):
        self.vs = load_vectorstore(
            db_type=db_type,
            save_path=str(settings.index_path),
            model_name=settings.embedding_model,
        )

    def retrieve(self, query: str, k: int = None):
        k = k or settings.top_k
        retriever = self.vs.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        # Return docs and scores if needed
        return docs
