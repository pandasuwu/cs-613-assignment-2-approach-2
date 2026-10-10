from pathlib import Path
from typing import Any

import polars as pl

from src.tables.common import get_mean_scalar, save_table_csv
from src.tables.schema import DATASET_APPROX_TOKENS


def build_table_9_context(master_lf: pl.LazyFrame, out_dir: Path) -> pl.DataFrame:
    """Generate Table 9: Sequence length and context scale impact on baseline quality and post-processing recovery."""
    master_df = master_lf.collect()

    ordered_datasets = [
        ("SICK-R", "similarity", "spearman_rho"),
        ("STSBenchmark", "similarity", "spearman_rho"),
        ("FiQA2018", "retrieval", "ndcg_at_10"),
        ("SCIDOCS", "retrieval", "ndcg_at_10"),
        ("ArguAna", "retrieval", "ndcg_at_10"),
        ("STS22.v2", "similarity", "spearman_rho"),
    ]

    rows: list[dict[str, Any]] = []

    for d, track, metric in ordered_datasets:
        sub = master_df.filter(
            (pl.col("dataset") == d)
            & (pl.col("track") == track)
            & (pl.col("regime") == "full")
            & (pl.col("metric") == metric)
        )

        ded_base = get_mean_scalar(
            sub.filter(
                (pl.col("tier") == "embedding") & (pl.col("method") == "baseline")
            )
        )

        base_mean_base = get_mean_scalar(
            sub.filter(
                (pl.col("tier") == "base")
                & (pl.col("pooling") == "mean_pooling")
                & (pl.col("method") == "baseline")
            )
        )

        base_last_base = get_mean_scalar(
            sub.filter(
                (pl.col("tier") == "base")
                & (pl.col("pooling") == "last_token_pooling")
                & (pl.col("method") == "baseline")
            )
        )

        soft_zca = get_mean_scalar(
            sub.filter(
                (pl.col("tier") == "base")
                & (pl.col("pooling") == "mean_pooling")
                & (pl.col("method") == "soft_zca")
            )
        )

        rows.append(
            {
                "dataset": d,
                "track": track,
                "metric": metric,
                "approx_mean_tokens": DATASET_APPROX_TOKENS[d],
                "dedicated_baseline": ded_base,
                "base_mean_baseline": base_mean_base,
                "base_last_baseline": base_last_base,
                "pooling_penalty": base_last_base - base_mean_base,
                "soft_zca_recovered_score": soft_zca,
                "recovery_gain": soft_zca - base_mean_base,
            }
        )

    result_df = pl.DataFrame(rows)
    out_file = out_dir / "table_9_context_length_impact.csv"
    save_table_csv(result_df, out_file, precision=4)
    return result_df
