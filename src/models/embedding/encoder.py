import torch
from sentence_transformers import SentenceTransformer

from src.config import logger


def encode_embedding_queries(
    texts: list[str],
    model: SentenceTransformer,
    model_id: str,
    instruction: str = "",
    batch_size: int = 32,
    show_progress: bool = True,
) -> torch.Tensor:
    """Encode retrieval queries using model-mandated asymmetric instruction prompts.

    Qwen3-Embedding uses instruction-tuned query formatting: 'Instruct: {instruction}\\nQuery: '.
    If no task instruction is provided, passing prompt=None triggers SentenceTransformer to use
    Qwen's built-in default web search query prompt.
    EmbeddingGemma uses its built-in query prompt ('task: search result | query: ') via encode_query.
    """
    logger.info("Encoding %d queries with model '%s'", len(texts), model_id)

    prompt = (
        f"Instruct: {instruction}\nQuery: "
        if ("qwen" in model_id.lower() and instruction)
        else None
    )

    embeddings = model.encode_query(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return embeddings.detach().cpu().to(torch.float32)


def encode_embedding_corpus(
    texts: list[str],
    model: SentenceTransformer,
    model_id: str,
    batch_size: int = 16,
    show_progress: bool = True,
) -> torch.Tensor:
    """Encode corpus documents using model-mandated document representations.

    SentenceTransformer.encode_document automatically selects the model's official document prompt:
    'title: none | text: ' for EmbeddingGemma, and '' (unprompted raw document) for Qwen3-Embedding.
    """
    logger.info("Encoding %d documents with model '%s'", len(texts), model_id)

    embeddings = model.encode_document(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return embeddings.detach().cpu().to(torch.float32)


def encode_embedding_sentences(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
) -> torch.Tensor:
    """Encode symmetric STS sentences plain without retrieval prompts."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_tensor=True,
    )
    return embeddings.detach().cpu().to(torch.float32)
