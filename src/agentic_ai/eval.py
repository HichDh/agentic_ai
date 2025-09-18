from __future__ import annotations

import random
import re
from typing import Dict, List

from datasets import load_dataset

from .agent import Agent
from .retriever import Retriever


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def _has_overlap(answer: str, contexts: List[str], n: int = 5) -> bool:
    toks_a = _tokenize(answer)
    grams_a = set(tuple(toks_a[i : i + n]) for i in range(len(toks_a) - n + 1))
    for ctx in contexts:
        toks_c = _tokenize(ctx)
        grams_c = set(tuple(toks_c[i : i + n]) for i in range(len(toks_c) - n + 1))
        if grams_a & grams_c:
            return True
    return False


def evaluate(
    dataset: str = "hotpot_qa",
    split: str = "validation",
    limit: int = 50,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    random.seed(seed)
    try:
        Retriever().load()
    except Exception:
        Retriever().index_folder("data/raw")
    agent = Agent()
    if dataset == "hotpot_qa":
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        idxs = random.sample(range(len(ds)), min(limit, len(ds)))
        questions = [ds[i]["question"] for i in idxs]
    elif dataset == "squad_v2":
        ds = load_dataset("squad_v2", split=split)
        idxs = random.sample(range(len(ds)), min(limit, len(ds)))
        questions = [ds[i]["question"] for i in idxs]
    else:
        raise ValueError("Unsupported dataset")
    nonempty = grounded = cited = 0
    for q in questions:
        res = agent.ask(q, k=k)
        ans = (res.get("answer") or "").strip()
        srcs = res.get("sources") or []
        ctxs = [s["text"] for s in srcs]
        if ans:
            nonempty += 1
        if ans and _has_overlap(ans, ctxs, n=5):
            grounded += 1
        if "[Source " in ans:
            cited += 1
    total = len(questions)
    return {
        "total": total,
        "nonempty_rate": nonempty / total if total else 0.0,
        "grounded_like_rate": grounded / total if total else 0.0,
        "cited_style_rate": cited / total if total else 0.0,
    }
