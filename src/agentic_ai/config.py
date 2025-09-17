from pathlib import Path
from pydantic import BaseModel, Field

import os

class Settings(BaseModel):
    # Paths
    data_dir: Path = Field(default=Path("data"))
    index_path: Path = Field(default=Path("data/index.faiss"))
    store_meta_path: Path = Field(default=Path("data/index_meta.json"))
    # Models
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    generator_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cpu"  # "cuda" if available
    # Retrieval
    top_k: int = 5
    backend: str = os.getenv("BACKEND", "local")
    system_prompt: str = os.getenv("SYSTEM_PROMPT", "You are a precise assistant. Use the provided sources. Cite as [Source i].")
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

settings = Settings()
