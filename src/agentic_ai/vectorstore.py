from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss
import numpy as np


@dataclass
class Doc:
    id: str
    text: str
    meta: Dict


class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.docs: List[Doc] = []

    def add(self, embeddings: np.ndarray, docs: List[Doc]):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.docs.extend(docs)

    def search(self, query_emb: np.ndarray, k: int) -> List[Tuple[Doc, float]]:
        faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb.astype(np.float32), k)
        results: List[Tuple[Doc, float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            results.append((self.docs[idx], float(score)))
        return results

    def save(self, path: str, meta_path: str):
        faiss.write_index(self.index, path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([d.__dict__ for d in self.docs], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, meta_path: str) -> "FaissStore":
        index = faiss.read_index(path)
        with open(meta_path, "r", encoding="utf-8") as f:
            docs_json = json.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.docs = [Doc(**d) for d in docs_json]
        return store
