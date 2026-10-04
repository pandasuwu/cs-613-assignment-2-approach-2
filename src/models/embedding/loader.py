import torch
from sentence_transformers import SentenceTransformer

from src.config import MAX_SEQ_LENGTH, get_device, logger


def load_embedding_model(model_id: str) -> SentenceTransformer:
    """Load a contrastively pre-trained SentenceTransformer embedding model."""
    device = str(get_device())
    logger.info("Loading embedding model '%s' onto device '%s'", model_id, device)

    model = SentenceTransformer(
        model_id,
        device=device,
        model_kwargs={"dtype": torch.float32},
    )
    model.to(torch.float32)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model
