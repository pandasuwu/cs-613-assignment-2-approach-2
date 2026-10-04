import sys
from pathlib import Path

# Add project root to sys.path so 'src' is resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import json
from typing import Any

import torch

from src.config import DEFAULT_BATCH_SIZE_DOCS, DEFAULT_BATCH_SIZE_QUERIES, logger
from src.data import load_retrieval_dataset, load_similarity_dataset
from src.models import (
    encode_base_texts,
    encode_embedding_corpus,
    encode_embedding_queries,
    encode_embedding_sentences,
    load_base_model,
    load_embedding_model,
)
from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    EMBEDDING_MODELS,
    RETRIEVAL_TASKS,
    SIMILARITY_TASKS,
    get_raw_cache_dir,
    is_base_model,
)


def encode_single_combination(
    model_id: str,
    task_name: str,
    pooling: str | None = None,
    overwrite: bool = False,
    batch_size_docs: int = DEFAULT_BATCH_SIZE_DOCS,
    batch_size_queries: int = DEFAULT_BATCH_SIZE_QUERIES,
    loaded_model: Any = None,
    loaded_tokenizer: Any = None,
) -> None:
    """Encode texts for a single model and task combination, saving FP32 tensors to raw cache."""
    cache_dir = get_raw_cache_dir(task_name, model_id, pooling=pooling)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_file = cache_dir / "meta.json"
    retrieval = task_name in RETRIEVAL_TASKS

    # Check for existing cached artifacts to ensure idempotent execution
    t1_file = cache_dir / ("corpus.pt" if retrieval else "sentences1.pt")
    t2_file = cache_dir / ("queries.pt" if retrieval else "sentences2.pt")
    if t1_file.exists() and t2_file.exists() and meta_file.exists() and not overwrite:
        logger.info(
            "Skipping %s on %s (already cached in %s)", model_id, task_name, cache_dir
        )
        return

    logger.info("Encoding %s on %s (pooling=%s)", model_id, task_name, pooling)

    if retrieval:
        ds_ret = load_retrieval_dataset(task_name)
        if is_base_model(model_id):
            base_model, base_tokenizer = (
                (loaded_model, loaded_tokenizer)
                if loaded_model
                else load_base_model(model_id)
            )
            pool_mode = "last_token" if pooling == "last_token_pooling" else "mean"
            t1 = encode_base_texts(
                ds_ret.doc_texts,
                base_model,
                base_tokenizer,
                pooling=pool_mode,
                batch_size=batch_size_docs,
            )
            t2 = encode_base_texts(
                ds_ret.query_texts,
                base_model,
                base_tokenizer,
                pooling=pool_mode,
                batch_size=batch_size_queries,
            )
        else:
            emb_model = loaded_model if loaded_model else load_embedding_model(model_id)
            t1 = encode_embedding_corpus(
                ds_ret.doc_texts, emb_model, model_id, batch_size=batch_size_docs
            )
            t2 = encode_embedding_queries(
                ds_ret.query_texts,
                emb_model,
                model_id,
                instruction=ds_ret.instruction,
                batch_size=batch_size_queries,
            )

        meta = {"qids": ds_ret.query_ids, "dids": ds_ret.doc_ids, "qrels": ds_ret.qrels}
    else:
        ds_sim = load_similarity_dataset(task_name)
        if is_base_model(model_id):
            base_model, base_tokenizer = (
                (loaded_model, loaded_tokenizer)
                if loaded_model
                else load_base_model(model_id)
            )
            pool_mode = "last_token" if pooling == "last_token_pooling" else "mean"
            t1 = encode_base_texts(
                ds_sim.sentences1,
                base_model,
                base_tokenizer,
                pooling=pool_mode,
                batch_size=batch_size_queries,
            )
            t2 = encode_base_texts(
                ds_sim.sentences2,
                base_model,
                base_tokenizer,
                pooling=pool_mode,
                batch_size=batch_size_queries,
            )
        else:
            emb_model = loaded_model if loaded_model else load_embedding_model(model_id)
            t1 = encode_embedding_sentences(
                ds_sim.sentences1, emb_model, batch_size=batch_size_queries
            )
            t2 = encode_embedding_sentences(
                ds_sim.sentences2, emb_model, batch_size=batch_size_queries
            )

        meta = {"scores": ds_sim.scores}

    torch.save(t1, t1_file)
    torch.save(t2, t2_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    logger.info("Saved raw cached representations to %s", cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Encode and cache raw representations."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (e.g. google/embeddinggemma-300m). Default: all models.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (e.g. FiQA2018, STSBenchmark). Default: all core tasks.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        help="Pooling mode for base LLMs (mean_pooling or last_token_pooling).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode and overwrite existing cache.",
    )
    parser.add_argument(
        "--batch-size-docs",
        type=int,
        default=DEFAULT_BATCH_SIZE_DOCS,
        help="Batch size for encoding documents.",
    )
    parser.add_argument(
        "--batch-size-queries",
        type=int,
        default=DEFAULT_BATCH_SIZE_QUERIES,
        help="Batch size for encoding queries/sentences.",
    )

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    tasks_to_run = [args.task] if args.task else (RETRIEVAL_TASKS + SIMILARITY_TASKS)

    for model_id in models_to_run:
        # Load model once outside task loop to avoid reloading weights repeatedly
        if is_base_model(model_id):
            loaded_model, loaded_tokenizer = load_base_model(model_id)
            poolings = [args.pooling] if args.pooling else BASE_POOLING_MODES
        else:
            loaded_model = load_embedding_model(model_id)
            loaded_tokenizer = None
            poolings = [None]

        for task_name in tasks_to_run:
            for pool_mode in poolings:
                encode_single_combination(
                    model_id=model_id,
                    task_name=task_name,
                    pooling=pool_mode,
                    overwrite=args.overwrite,
                    batch_size_docs=args.batch_size_docs,
                    batch_size_queries=args.batch_size_queries,
                    loaded_model=loaded_model,
                    loaded_tokenizer=loaded_tokenizer,
                )

        # Offload model to CPU and clear accelerator cache to prevent memory contention between models
        if loaded_model is not None:
            if hasattr(loaded_model, "to"):
                loaded_model.to("cpu")
            del loaded_model, loaded_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
