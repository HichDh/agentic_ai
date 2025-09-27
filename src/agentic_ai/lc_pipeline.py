from langchain_community.vectorstores import FAISS, Chroma  # Add more as needed
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from pathlib import Path
from numbers import Number
from operator import itemgetter

VECTOR_DB_MAP = {
    "faiss": FAISS,
    "chroma": Chroma,
    # Add more vector DBs here
}

def build_vectorstore(
    folder="data/raw",
    db_type="faiss",
    model_name="sentence-transformers/all-mpnet-base-v2",
    save_path="data/faiss_index"
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=120)
    docs = []
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in {".txt", ".md"}:
            docs.append({"page_content": p.read_text(encoding="utf-8", errors="ignore"), "metadata":{"path":str(p)}})
    chunks = []
    for d in docs:
        for c in splitter.split_text(d["page_content"]):
            chunks.append({"page_content": c, "metadata": d["metadata"]})
    embed = HuggingFaceEmbeddings(model_name=model_name)
    VectorDB = VECTOR_DB_MAP[db_type]
    if db_type == "faiss":
        vs = VectorDB.from_texts([c["page_content"] for c in chunks], embed, metadatas=[c["metadata"] for c in chunks])
        vs.save_local(save_path)
    elif db_type == "chroma":
        vs = VectorDB.from_texts([c["page_content"] for c in chunks], embed, metadatas=[c["metadata"] for c in chunks], persist_directory=save_path)
        vs.persist()
    # Add more DBs as needed
    return vs

def load_vectorstore(
    db_type="faiss",
    save_path="data/faiss_index",
    model_name="sentence-transformers/all-mpnet-base-v2"
):
    embed = HuggingFaceEmbeddings(model_name=model_name)
    VectorDB = VECTOR_DB_MAP[db_type]
    if db_type == "faiss":
        return VectorDB.load_local(save_path, embed)
    elif db_type == "chroma":
        return VectorDB(persist_directory=save_path, embedding_function=embed)
    # Add more DBs as needed

def build_rag_chain(vs, llm=None, k=5, min_score: float = 0.30, strict: bool = True):
    def _retrieve_with_scores(q: str):
        raw = vs.similarity_search_with_relevance_scores(q, k=max(20, k * 4))
        kept = []
        for doc, score in raw:
            s = float(score)
            doc.metadata = {**(doc.metadata or {}), "relevance_score": s}
            if s >= min_score:
                kept.append(doc)
        if strict and not kept:
            return []
        return kept[:k]

    scored_retriever = RunnableLambda(_retrieve_with_scores)

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are precise. Only answer from the context. Cite as [Source i]. If not found, say you don't know."),
    #     MessagesPlaceholder("chat_history"),
    #     ("human", "Question: {question}\n\nContext:\n{context}")
    # ])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are precise. Use the retrieved context for external knowledge questions. "
        "You may use chat history for questions about the conversation itself "
        "Cite sources as [Source i] only when you use the retrieved context. "
        "If neither context nor history contains the answer, say you don't know."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    llm = llm or ChatHuggingFace(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def _src_path(md):
        return md.get("source") or md.get("path") or md.get("file_path") or md.get("document_id") or "?"

    def format_docs(docs):
        parts = []
        for i, d in enumerate(docs, 1):
            path = _src_path(d.metadata)
            page = d.metadata.get("page")
            score = d.metadata.get("relevance_score")
            score_str = f" | score={float(score):.3f}" if isinstance(score, Number) else ""
            loc = f"{path}" + (f", p.{page}" if page is not None else "")
            snippet = d.page_content.strip().replace("\n", " ")
            if len(snippet) > 600:
                snippet = snippet[:600] + "..."
            parts.append(f"[Source {i}] {snippet}\n({loc}{score_str})")
        return "\n\n".join(parts) if parts else ""

    qa_chain = (
        {
            "context": itemgetter("question") | scored_retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
    )

    return RunnableParallel(
        answer=qa_chain,
        source_documents=itemgetter("question") | scored_retriever,
    )
