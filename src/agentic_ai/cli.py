from __future__ import annotations

import typer
from rich import print

from .agent import Agent
from .datasets_download import download as _download
from .eval import evaluate as _evaluate
from .retriever import Retriever

app = typer.Typer(help="Agentic RAG CLI")


@app.command()
def index(
    folder: str = typer.Option("data/raw", help="Folder containing docs to index")
):
    r = Retriever()
    r.index_folder(folder)
    print(":white_check_mark: [bold]Indexed[/bold]")


@app.command()
def ask(question: str, k: int = 5):
    agent = Agent()
    res = agent.ask(question, k=k)
    print("\n[bold]Answer[/bold]:", res["answer"])
    print("\n[bold]Sources[/bold]:")
    for i, s in enumerate(res["sources"], 1):
        print(f"[{i}] {s['meta'].get('path', '?')} (score={s['score']:.3f})")


@app.command()
def download(
    dataset: str = typer.Option("hotpot_qa", help="hotpot_qa | squad_v2"),
    split: str = typer.Option("train"),
    limit: int = typer.Option(1000, help="Max examples to ingest as docs"),
):
    """Download a public dataset into data/raw/<dataset>/ (no API keys)."""
    _download(dataset, split, limit)


@app.command()
def eval(
    dataset: str = typer.Option("hotpot_qa", help="hotpot_qa | squad_v2"),
    split: str = typer.Option("validation"),
    limit: int = typer.Option(50),
    k: int = typer.Option(5),
):
    """Lightweight evaluation to approximate groundedness & citation discipline."""
    metrics = _evaluate(dataset=dataset, split=split, limit=limit, k=k)
    print(metrics)
