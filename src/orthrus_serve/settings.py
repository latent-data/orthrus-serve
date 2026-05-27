"""Single source of truth for env-var configuration.

Loaded once on import. Modules that previously called `os.environ.get(...)`
import `settings` from here and read attributes off the dataclass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_ORTHRUS_REVISION = "977a617772e91c966a8cd9b551f4151f9824b6fa"
_DEFAULT_QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


@dataclass(frozen=True)
class Settings:
    debug: bool
    base_model: bool
    enable_thinking: bool | None  # None = let the chat template default decide
    diffusion_enabled: bool
    orthrus_revision: str
    qwen_revision: str
    quant: str | None  # one of orthrus_serve.quantization.SUPPORTED_QUANT_SCHEMES,
                       # or None for bf16 (no quantization)

    @property
    def served_model_id(self) -> str:
        return "qwen3-8b" if self.base_model else "orthrus-qwen3-8b"

    @classmethod
    def from_env(cls) -> Settings:
        thinking_env = os.environ.get("ORTHRUS_ENABLE_THINKING")
        thinking = None if thinking_env is None else (thinking_env == "true")
        base = os.environ.get("ORTHRUS_BASE_MODEL", "0") == "1"
        return cls(
            debug=os.environ.get("ORTHRUS_DEBUG", "0") == "1",
            base_model=base,
            enable_thinking=thinking,
            diffusion_enabled=(
                not base and os.environ.get("ORTHRUS_DIFFUSION", "1") != "0"
            ),
            orthrus_revision=os.environ.get("ORTHRUS_REVISION", _DEFAULT_ORTHRUS_REVISION),
            qwen_revision=os.environ.get("QWEN_REVISION", _DEFAULT_QWEN_REVISION),
            quant=os.environ.get("ORTHRUS_QUANT") or None,
        )


settings: Settings = Settings.from_env()
