from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import MAX_SEQ_LENGTH, logger
from src.models.base.pooling import last_token_pooling, mean_pooling


def encode_base_texts(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    pooling: str = "mean",
    batch_size: int = 16,
    show_progress: bool = True,
) -> torch.Tensor:
    """Encode text strings into L2-normalized FP32 representations using a base language model."""
    if pooling not in ["mean", "last_token"]:
        raise ValueError(
            f"Unsupported pooling mode: '{pooling}'. Must be 'mean' or 'last_token'."
        )

    device = next(model.parameters()).device
    padding_side = getattr(tokenizer, "padding_side", "right")

    embeddings_list: list[torch.Tensor] = []
    num_texts = len(texts)
    iterator = range(0, num_texts, batch_size)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc=f"Encoding (base, {pooling})",
            total=(num_texts + batch_size - 1) // batch_size,
        )

    for i in iterator:
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

        if pooling == "mean":
            pooled = mean_pooling(hidden_states, attention_mask)
        else:
            pooled = last_token_pooling(
                hidden_states, attention_mask, padding_side=padding_side
            )

        normalized = F.normalize(pooled, p=2, dim=1)
        embeddings_list.append(normalized.detach().cpu().to(torch.float32))

        # Explicitly release GPU tensor references to prevent MPS allocator memory accumulation
        del outputs, hidden_states, pooled, normalized, input_ids, attention_mask
        if (i // batch_size) % 50 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    result = torch.cat(embeddings_list, dim=0)
    logger.info(
        "Encoded %d texts -> tensor shape %s (FP32)", len(texts), tuple(result.shape)
    )
    return result
