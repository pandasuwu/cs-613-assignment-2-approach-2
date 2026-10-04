from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import MAX_SEQ_LENGTH, get_device, logger


def load_base_model(model_id: str) -> tuple[Any, Any]:
    """Load an un-fine-tuned base causal language model and tokenizer onto target accelerator."""
    device = get_device()
    logger.info("Initializing base LLM '%s' onto device '%s'", model_id, device)

    logger.info("Fetching tokenizer for '%s'...", model_id)
    tokenizer: Any = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    logger.info("Fetching model weights for '%s' (initial download may take 1-3 minutes)...", model_id)
    model: Any = AutoModel.from_pretrained(
        model_id,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.to(device=device, dtype=torch.float32)
    model.eval()
    logger.info("Model '%s' loaded successfully onto '%s'.", model_id, device)

    return model, tokenizer
