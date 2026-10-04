import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.config import DEFAULT_BATCH_SIZE_SEARCH, get_device, logger
from src.metrics import compute_retrieval_metrics
from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    COMPRESSION_LADDER_K,
    COMPRESSION_METHODS,
    EMBEDDING_MODELS,
    FULL_METHODS,
    RETRIEVAL_TASKS,
    get_raw_cache_dir,
    get_result_dir,
    get_transformed_cache_dir,
    is_base_model,
    write_result_csv,
)


def evaluate_retrieval_for_method(
    corpus: torch.Tensor,
    queries: torch.Tensor,
    qids: list[str],
    dids: list[str],
    qrels: dict[str, dict[str, int]],
    out_csv: Path,
    batch_size_search: int = DEFAULT_BATCH_SIZE_SEARCH,
    extra_cols: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics (nDCG@10, Recall@100, MRR@10) and save atomic CSV."""
    num_queries = queries.shape[0]
    run: dict[str, dict[str, float]] = {}

    device = get_device()
    corpus_dev = corpus.to(device)

    # Batched top-100 similarity search
    for i in range(0, num_queries, batch_size_search):
        q_batch = queries[i : i + batch_size_search].to(device)
        scores_batch = torch.matmul(q_batch, corpus_dev.T)  # (batch_size, num_docs)
        topk_scores, topk_indices = torch.topk(
            scores_batch, k=min(100, corpus.shape[0]), dim=1
        )
        topk_scores_cpu = topk_scores.cpu()
        topk_indices_cpu = topk_indices.cpu()

        for b_idx in range(q_batch.shape[0]):
            qid = qids[i + b_idx]
            run[qid] = {
                dids[idx.item()]: float(score.item())
                for idx, score in zip(topk_indices_cpu[b_idx], topk_scores_cpu[b_idx])
            }

    metrics = compute_retrieval_metrics(qrels, run)
    write_result_csv(out_csv, metrics, extra_cols=extra_cols, precision=6)
    return metrics


def evaluate_retrieval_combination(
    model_id: str,
    task_name: str,
    pooling: str | None = None,
    overwrite: bool = False,
    batch_size_search: int = DEFAULT_BATCH_SIZE_SEARCH,
) -> None:
    """Evaluate retrieval metrics across all full and compression transforms for a combination."""
    raw_dir = get_raw_cache_dir(task_name, model_id, pooling=pooling)
    meta_file = raw_dir / "meta.json"
    if not meta_file.exists():
        logger.warning("Metadata not found in %s. Run encode.py first.", raw_dir)
        return

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    qids, dids, qrels = meta["qids"], meta["dids"], meta["qrels"]

    # 1. Full-Dimension Transforms
    for method in FULL_METHODS:
        csv_path = (
            get_result_dir("retrieval", task_name, model_id, "full", pooling=pooling)
            / f"{method}.csv"
        )
        if csv_path.exists() and not overwrite:
            continue

        trans_dir = get_transformed_cache_dir(
            task_name, model_id, "full", method, pooling=pooling
        )
        c_file = trans_dir / "corpus.pt"
        q_file = trans_dir / "queries.pt"
        if not (c_file.exists() and q_file.exists()):
            continue

        corpus = torch.load(c_file, weights_only=True)
        queries = torch.load(q_file, weights_only=True)

        res = evaluate_retrieval_for_method(
            corpus,
            queries,
            qids,
            dids,
            qrels,
            csv_path,
            batch_size_search=batch_size_search,
        )
        logger.info(
            "[%s | %s | full | %s] nDCG@10=%.4f  Recall@100=%.4f  MRR@10=%.4f",
            task_name,
            model_id,
            method,
            res["ndcg_at_10"],
            res["recall_at_100"],
            res["mrr_at_10"],
        )

    # 2. Compression Transforms
    for method in COMPRESSION_METHODS:
        for k in COMPRESSION_LADDER_K:
            sub_name = f"{method}_k{k}"
            csv_path = (
                get_result_dir(
                    "retrieval", task_name, model_id, "compression", pooling=pooling
                )
                / f"{sub_name}.csv"
            )
            if csv_path.exists() and not overwrite:
                continue

            trans_dir = get_transformed_cache_dir(
                task_name, model_id, "compression", sub_name, pooling=pooling
            )
            c_file = trans_dir / "corpus.pt"
            q_file = trans_dir / "queries.pt"
            if not (c_file.exists() and q_file.exists()):
                continue

            corpus = torch.load(c_file, weights_only=True)
            queries = torch.load(q_file, weights_only=True)

            gamma_val: float | None = None
            if (trans_dir / "gamma.json").exists():
                with open(trans_dir / "gamma.json", encoding="utf-8") as gf:
                    gamma_val = json.load(gf).get("gamma")

            extra = {
                "k": k,
                "gamma": f"{gamma_val:.4f}" if gamma_val is not None else "",
            }
            res = evaluate_retrieval_for_method(
                corpus,
                queries,
                qids,
                dids,
                qrels,
                csv_path,
                batch_size_search=batch_size_search,
                extra_cols=extra,
            )
            logger.info(
                "[%s | %s | compression | %s] nDCG@10=%.4f  Recall@100=%.4f  MRR@10=%.4f",
                task_name,
                model_id,
                sub_name,
                res["ndcg_at_10"],
                res["recall_at_100"],
                res["mrr_at_10"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Information Retrieval evaluation (nDCG@10, Recall@100, MRR@10)."
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model ID. Default: all models."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name. Default: all retrieval tasks.",
    )
    parser.add_argument(
        "--pooling", type=str, default=None, help="Pooling mode for base LLMs."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate and overwrite existing CSVs.",
    )
    parser.add_argument(
        "--batch-size-search",
        type=int,
        default=DEFAULT_BATCH_SIZE_SEARCH,
        help="Batch size for similarity ranking.",
    )

    args = parser.parse_args()

    models_to_run = [args.model] if args.model else (EMBEDDING_MODELS + BASE_MODELS)
    tasks_to_run = [args.task] if args.task else RETRIEVAL_TASKS

    for model_id in models_to_run:
        poolings = (
            [args.pooling]
            if args.pooling
            else (BASE_POOLING_MODES if is_base_model(model_id) else [None])
        )
        for task_name in tasks_to_run:
            for pool_mode in poolings:
                evaluate_retrieval_combination(
                    model_id=model_id,
                    task_name=task_name,
                    pooling=pool_mode,
                    overwrite=args.overwrite,
                    batch_size_search=args.batch_size_search,
                )


if __name__ == "__main__":
    main()
