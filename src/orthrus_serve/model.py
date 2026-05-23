from __future__ import annotations

import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

ORTHRUS_MODEL_ID = "chiennv/Orthrus-Qwen3-8B"
QWEN_MODEL_ID = "Qwen/Qwen3-8B"


def load_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    use_base_model = os.environ.get("ORTHRUS_BASE_MODEL", "0") == "1"
    qwen_revision = os.environ.get(
        "QWEN_REVISION", "b968826d9c46dd6066d109eabc6255188de91218"
    )

    if use_base_model:
        logger.info("BASE MODEL MODE: loading %s (revision=%s)", QWEN_MODEL_ID, qwen_revision)
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, revision=qwen_revision)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            revision=qwen_revision,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).eval()
        logger.info("Base model loaded on %s", next(model.parameters()).device)
        return model, tokenizer

    orthrus_revision = os.environ.get(
        "ORTHRUS_REVISION", "977a617772e91c966a8cd9b551f4151f9824b6fa"
    )

    logger.info("Loading Orthrus tokenizer (revision=%s)", orthrus_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        ORTHRUS_MODEL_ID,
        revision=orthrus_revision,
        trust_remote_code=True,
    )

    logger.info("Loading Orthrus model (revision=%s)", orthrus_revision)
    model = AutoModelForCausalLM.from_pretrained(
        ORTHRUS_MODEL_ID,
        revision=orthrus_revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()

    logger.info("Model loaded successfully on %s", next(model.parameters()).device)
    return model, tokenizer
