import sys
from pathlib import Path

# Add project root to sys.path so 'src' is resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Enforce project-local cache isolation before importing huggingface_hub
from src.config import HF_HOME, logger  # isort: skip

import argparse
import os

from huggingface_hub import snapshot_download

from src.data import load_retrieval_dataset, load_similarity_dataset
from src.registry import (
    BASE_MODELS,
    EMBEDDING_MODELS,
    RETRIEVAL_TASKS,
    SIMILARITY_TASKS,
)


def download_model(model_id: str) -> None:
    """Download full model checkpoint into project-local cache with progress tracking."""
    hub_cache = HF_HOME / "hub"
    logger.info("Downloading checkpoint: %s (target cache: %s)", model_id, hub_cache)
    path = snapshot_download(
        repo_id=model_id,
        token=os.environ.get("HF_TOKEN"),
        cache_dir=str(hub_cache),
    )
    logger.info("Successfully cached %s -> %s", model_id, path)


def download_dataset(task_name: str) -> None:
    """Download, cache, and verify an MTEB v2 benchmark dataset."""
    logger.info("Downloading and verifying dataset: %s", task_name)
    if task_name in RETRIEVAL_TASKS:
        ds_ret = load_retrieval_dataset(task_name)
        logger.info(
            "Successfully cached %s (%d docs, %d queries, %d qrels)",
            task_name,
            len(ds_ret.doc_texts),
            len(ds_ret.query_texts),
            len(ds_ret.qrels),
        )
    elif task_name in SIMILARITY_TASKS:
        ds_sim = load_similarity_dataset(task_name)
        logger.info(
            "Successfully cached %s (%d sentence pairs)",
            task_name,
            len(ds_sim.sentences1),
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download all evaluated models and MTEB v2 datasets into project-local cache."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model ID to download. Default: all 4 models.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task dataset to download. Default: all 7 benchmark datasets.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Download only model checkpoints.",
    )
    parser.add_argument(
        "--tasks-only",
        action="store_true",
        help="Download only task datasets.",
    )

    args = parser.parse_args()

    models_to_download = (
        [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    )
    tasks_to_download = (
        [args.task] if args.task else (RETRIEVAL_TASKS + SIMILARITY_TASKS)
    )

    if not args.tasks_only:
        for model_id in models_to_download:
            download_model(model_id)

    if not args.models_only:
        for task_name in tasks_to_download:
            download_dataset(task_name)

    logger.info("All requested assets are downloaded and verified in local cache.")


if __name__ == "__main__":
    main()
