from pydantic import BaseModel, Field
from pathlib import Path
import os

class Settings(BaseModel):
    # Paths
    data_dir: Path = Field(default=Path("data"))
    index_path: Path = Field(default=Path("data/index_faiss"))

    # Models
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    generator_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Runtime/device
    device: str = "auto"              # "auto" | "cpu" | "mps" | "cuda"
    torch_dtype: str = "float32"      # "float32" or "bfloat16" (bf16 faster, try later)
    attn_implementation: str | None = "sdpa"  # set to None if your stack complains

    # Generation defaults
    max_new_tokens: int = 192
    do_sample_default: bool = False
    temperature_default: float = 0.7
    use_cache: bool = True

    # Retrieval
    top_k: int = 3
    backend: str = os.getenv("BACKEND", "local")
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a precise assistant. Use the provided sources. Cite as [Source i].",
    )

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
