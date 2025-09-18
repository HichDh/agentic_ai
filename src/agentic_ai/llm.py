from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import settings


class LocalGenerator:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.generator_model
        self.device = device or settings.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def chat(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return (
            text.split("<|assistant|>")[-1].strip() if "<|assistant|>" in text else text
        )


class OpenAIBackend:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        # Using the Responses API (new SDK)
        rsp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=max_new_tokens,
        )
        return rsp.choices[0].message.content


class AnthropicBackend:
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        msg = self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_new_tokens,
            messages=[{"role": "user", "content": user}],
            temperature=0.3,
        )
        return "".join([b.text for b in msg.content if b.type == "text"])


class ProviderRouter:
    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or settings.backend).lower()
        if self.backend == "openai":
            self.impl = OpenAIBackend()
        elif self.backend == "anthropic":
            self.impl = AnthropicBackend()
        else:
            self.impl = LocalGenerator()

    def generate(self, system: str, user: str) -> str:
        return self.impl.chat(system, user)
