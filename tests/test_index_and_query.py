from pathlib import Path

from agentic_ai.retriever import Retriever


def test_index_and_retrieve(tmp_path: Path):
    # Create toy doc
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text(
        "Agentic RAG uses retrieval and planning.", encoding="utf-8"
    )
    r = Retriever()
    r.index_folder(str(d))
    hits = r.retrieve("What does agentic RAG use?")
    assert len(hits) > 0
