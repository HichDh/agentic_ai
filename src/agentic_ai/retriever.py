from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import re
from .embeddings import Embedder
from .vectorstore import FaissStore, Doc
from .config import settings

def _read_textlike(path: Path) -> str:
    text = ""
    if path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() == ".pdf":
        from pdfminer.high_level import extract_text
        text = extract_text(str(path)) or ""
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
    return re.sub(r"\s+", " ", text).strip()

class Retriever:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.store: FaissStore | None = None

    def index_folder(self, folder: str | Path):
        folder = Path(folder)
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".txt",".md",".pdf"}]
        texts = [_read_textlike(p) for p in files]
        metas = [{"path": str(p)} for p in files]
        docs = [Doc(id=str(i), text=t, meta=m) for i,(t,m) in enumerate(zip(texts, metas)) if t]
        embeddings = self.embedder.encode([d.text for d in docs])
        self.store = FaissStore(dim=embeddings.shape[1])
        self.store.add(embeddings, docs)
        self.store.save(str(settings.index_path), str(settings.store_meta_path))

    def load(self):
        self.store = FaissStore.load(str(settings.index_path), str(settings.store_meta_path))

    def retrieve(self, query: str, k: int | None = None) -> List[Tuple[Doc, float]]:
        if self.store is None:
            self.load()
        k = k or settings.top_k
        q_emb = self.embedder.encode([query])
        assert self.store is not None
        return self.store.search(q_emb, k=k)
