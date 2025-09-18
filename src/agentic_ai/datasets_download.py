# src/agentic_ai/datasets_download.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

from datasets import load_dataset
from rich import print

DATA_DIR = Path("data/raw")


def _sanitize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:60] or "doc"


def _write_docs(docs: Iterable[tuple[str, str]], prefix: str):
    out_dir = DATA_DIR / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, (title, text) in enumerate(docs):
        if not text:
            continue
        (out_dir / f"{i:05d}_{_safe_name(title)}.txt").write_text(
            text, encoding="utf-8"
        )
        count += 1
    print(f":white_check_mark: Wrote {count} docs to {out_dir}")


def download(
    dataset: Literal["hotpot_qa", "squad_v2"], split: str = "train", limit: int = 1000
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if dataset == "hotpot_qa":
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        n = min(limit, len(ds))
        docs = []
        for ex in ds.select(range(n)):
            ctx = ex["context"]
            # New schema: dict with parallel lists
            if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
                for title, sents in zip(ctx["title"], ctx["sentences"]):
                    text = _sanitize(" ".join(sents))
                    if text:
                        docs.append((title or "hotpot", text))
            # Old schema: list of (title, sentences) pairs
            elif isinstance(ctx, list):
                for title, sents in ctx:
                    text = _sanitize(" ".join(sents))
                    if text:
                        docs.append((title or "hotpot", text))
            else:
                # Fallback: skip unknown shapes
                continue
        _write_docs(docs, prefix="hotpot_qa")

    elif dataset == "squad_v2":
        ds = load_dataset("squad_v2", split=split)
        n = min(limit, len(ds))
        docs = []
        for ex in ds.select(range(n)):
            title = ex.get("title") or "article"
            context = _sanitize(ex["context"])
            if context:
                docs.append((title, context))
        _write_docs(docs, prefix="squad_v2")
    else:
        raise ValueError("Unsupported dataset")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hotpot_qa", "squad_v2"], default="hotpot_qa")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    download(args.dataset, args.split, args.limit)
