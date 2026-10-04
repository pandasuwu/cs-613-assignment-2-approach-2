import argparse
import os

from huggingface_hub import snapshot_download

from src.config import HF_HOME, logger
from src.data import load_retrieval_dataset, load_similarity_dataset
from src.registry import (
    BASE_MODELS,
    EMBEDDING_MODELS,
    RETRIEVAL_TASKS,
    SIMILARITY_TASKS,
)


def download_model(model_id: str) -> None:
    """Download full model checkpoint into project-local cache with progress tracking."""
    logger.info("Downloading checkpoint: %s (target cache: %s)", model_id, HF_HOME)
    snapshot_download(
        repo_id=model_id,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info("Successfully cached %s", model_id)


def download_dataset(task_name: str) -> None:
    """Download and cache an MTEB v2 benchmark dataset."""
    logger.info("Downloading dataset: %s", task_name)
    if task_name in RETRIEVAL_TASKS:
        load_retrieval_dataset(task_name)
    elif task_name in SIMILARITY_TASKS:
        load_similarity_dataset(task_name)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    logger.info("Successfully cached %s", task_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download all evaluated models and MTEB v2 datasets into local cache."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to download. Default: all 4 models.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task dataset to download. Default: all 7 benchmark datasets.",
    )

    args = parser.parse_args()

    models_to_download = [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    tasks_to_download = [args.task] if args.task else (RETRIEVAL_TASKS + SIMILARITY_TASKS)

    for model_id in models_to_download:
        download_model(model_id)

    for task_name in tasks_to_download:
        download_dataset(task_name)

    logger.info("All requested assets are downloaded and verified in local cache.")


if __name__ == "__main__":
    main()
