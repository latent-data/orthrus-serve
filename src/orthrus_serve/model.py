from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .quantization import apply_quantization
from .settings import settings

logger = logging.getLogger(__name__)

ORTHRUS_MODEL_ID = "chiennv/Orthrus-Qwen3-8B"
QWEN_MODEL_ID = "Qwen/Qwen3-8B"


def load_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if settings.base_model:
        logger.info("BASE MODEL MODE: loading %s (revision=%s)", QWEN_MODEL_ID, settings.qwen_revision)
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, revision=settings.qwen_revision)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            revision=settings.qwen_revision,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).eval()
        apply_quantization(model, settings.quant)
        logger.info("Base model loaded on %s (quant=%s)",
                    next(model.parameters()).device, settings.quant or "bf16")
        return model, tokenizer

    logger.info("Loading Orthrus tokenizer (revision=%s)", settings.orthrus_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        ORTHRUS_MODEL_ID,
        revision=settings.orthrus_revision,
        trust_remote_code=True,
    )

    logger.info("Loading Orthrus model (revision=%s)", settings.orthrus_revision)
    model = AutoModelForCausalLM.from_pretrained(
        ORTHRUS_MODEL_ID,
        revision=settings.orthrus_revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    apply_quantization(model, settings.quant)

    logger.info("Model loaded successfully on %s (quant=%s)",
                next(model.parameters()).device, settings.quant or "bf16")
    return model, tokenizer
