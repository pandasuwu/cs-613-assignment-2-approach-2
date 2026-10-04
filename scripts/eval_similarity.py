import sys
from pathlib import Path

# Add project root to sys.path so 'src' is resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Any

import torch

from src.config import get_device, logger
from src.metrics import compute_similarity_metrics
from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    COMPRESSION_LADDER_K,
    COMPRESSION_METHODS,
    EMBEDDING_MODELS,
    FULL_METHODS,
    SIMILARITY_TASKS,
    get_raw_cache_dir,
    get_result_dir,
    get_transformed_cache_dir,
    is_base_model,
    write_result_csv,
)


def evaluate_similarity_for_method(
    s1: torch.Tensor,
    s2: torch.Tensor,
    gold_scores: list[float],
    out_csv: Path,
    extra_cols: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute STS similarity metrics (Spearman rho, Pearson r) and save atomic CSV."""
    # Row-wise cosine similarity between paired sentences
    sims = torch.sum(s1 * s2, dim=1).cpu()

    metrics = compute_similarity_metrics(sims, gold_scores)
    write_result_csv(out_csv, metrics, extra_cols=extra_cols, precision=4)
    return metrics


def evaluate_similarity_combination(
    model_id: str,
    task_name: str,
    pooling: str | None = None,
    overwrite: bool = False,
) -> None:
    """Evaluate STS benchmark metrics across all full and compression transforms."""
    raw_dir = get_raw_cache_dir(task_name, model_id, pooling=pooling)
    meta_file = raw_dir / "meta.json"
    if not meta_file.exists():
        logger.warning("Metadata not found in %s. Run encode.py first.", raw_dir)
        return

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    gold_scores = meta["scores"]

    # 1. Full-Dimension Transforms
    for method in FULL_METHODS:
        csv_path = (
            get_result_dir("similarity", task_name, model_id, "full", pooling=pooling)
            / f"{method}.csv"
        )
        if csv_path.exists() and not overwrite:
            continue

        trans_dir = get_transformed_cache_dir(
            task_name, model_id, "full", method, pooling=pooling
        )
        s1_file = trans_dir / "sentences1.pt"
        s2_file = trans_dir / "sentences2.pt"
        if not (s1_file.exists() and s2_file.exists()):
            continue

        s1 = torch.load(s1_file, weights_only=True).to(get_device())
        s2 = torch.load(s2_file, weights_only=True).to(get_device())

        res = evaluate_similarity_for_method(s1, s2, gold_scores, csv_path)
        logger.info(
            "[%s | %s | full | %s] Spearman=%.2f  Pearson=%.2f",
            task_name,
            model_id,
            method,
            res["spearman_rho"],
            res["pearson_r"],
        )

    # 2. Compression Transforms
    for method in COMPRESSION_METHODS:
        for k in COMPRESSION_LADDER_K:
            sub_name = f"{method}_k{k}"
            csv_path = (
                get_result_dir(
                    "similarity", task_name, model_id, "compression", pooling=pooling
                )
                / f"{sub_name}.csv"
            )
            if csv_path.exists() and not overwrite:
                continue

            trans_dir = get_transformed_cache_dir(
                task_name, model_id, "compression", sub_name, pooling=pooling
            )
            s1_file = trans_dir / "sentences1.pt"
            s2_file = trans_dir / "sentences2.pt"
            if not (s1_file.exists() and s2_file.exists()):
                continue

            s1 = torch.load(s1_file, weights_only=True).to(get_device())
            s2 = torch.load(s2_file, weights_only=True).to(get_device())

            gamma_val: float | None = None
            if (trans_dir / "gamma.json").exists():
                with open(trans_dir / "gamma.json", encoding="utf-8") as gf:
                    gamma_val = json.load(gf).get("gamma")

            extra = {
                "k": k,
                "gamma": f"{gamma_val:.4f}" if gamma_val is not None else "",
            }
            res = evaluate_similarity_for_method(
                s1, s2, gold_scores, csv_path, extra_cols=extra
            )
            logger.info(
                "[%s | %s | compression | %s] Spearman=%.2f  Pearson=%.2f",
                task_name,
                model_id,
                sub_name,
                res["spearman_rho"],
                res["pearson_r"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Semantic Textual Similarity (STS) evaluation (Spearman rho, Pearson r)."
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model ID. Default: all models."
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Task name. Default: all STS tasks."
    )
    parser.add_argument(
        "--pooling", type=str, default=None, help="Pooling mode for base LLMs."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate and overwrite existing CSVs.",
    )

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    tasks_to_run = [args.task] if args.task else SIMILARITY_TASKS

    for model_id in models_to_run:
        poolings = (
            [args.pooling]
            if args.pooling
            else (BASE_POOLING_MODES if is_base_model(model_id) else [None])
        )
        for task_name in tasks_to_run:
            for pool_mode in poolings:
                evaluate_similarity_combination(
                    model_id=model_id,
                    task_name=task_name,
                    pooling=pool_mode,
                    overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()
