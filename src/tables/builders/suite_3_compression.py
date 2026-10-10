from pathlib import Path
from typing import Any

import polars as pl

from src.tables.common import save_table_csv
from src.tables.schema import (
    CANONICAL_MODEL_COLUMNS,
    IR_DATASETS,
    LADDER_K_ORDERED,
    METHODS_COMP_ORDERED,
    MODEL_EMBGEMMA,
    MODEL_QWEN_EMB,
    STS_DATASETS,
)

IR_METRIC_SHORT: dict[str, str] = {
    "ndcg_at_10": "ndcg",
    "recall_at_100": "recall",
    "mrr_at_10": "mrr",
}

STS_METRIC_SHORT: dict[str, str] = {
    "spearman_rho": "spearman",
    "pearson_r": "pearson",
}

DATASET_SHORT: dict[str, str] = {
    "FiQA2018": "fiqa",
    "ArguAna": "arguana",
    "SCIDOCS": "scidocs",
    "STSBenchmark": "stsb",
    "SICK-R": "sickr",
    "STS22.v2": "sts22",
}


def build_single_compression_table(
    master_df: pl.DataFrame,
    dataset: str,
    track: str,
    metric_name: str,
    out_file: Path,
    precision: int = 4,
) -> pl.DataFrame:
    """Build a single-purpose, granular compression table for one dataset and one metric."""
    # 1. Native baseline scores for this dataset and metric
    native_sub = master_df.filter(
        (pl.col("track") == track)
        & (pl.col("dataset") == dataset)
        & (pl.col("regime") == "full")
        & (pl.col("method") == "baseline")
        & (pl.col("metric") == metric_name)
    )
    native_map = dict(zip(native_sub["model"], native_sub["value"]))

    # 2. Compression runs for this dataset and metric
    comp_sub = master_df.filter(
        (pl.col("track") == track)
        & (pl.col("dataset") == dataset)
        & (pl.col("regime") == "compression")
        & (pl.col("metric") == metric_name)
    )

    rows: list[dict[str, Any]] = []

    for k in LADDER_K_ORDERED:
        for m in METHODS_COMP_ORDERED:
            row: dict[str, Any] = {
                "k": k,
                "method": m,
            }

            sub = comp_sub.filter((pl.col("k") == k) & (pl.col("method") == m))
            scores = dict(zip(sub["model"], sub["value"]))

            ded_retentions: list[float] = []
            base_retentions: list[float] = []

            for model_col in CANONICAL_MODEL_COLUMNS:
                score = scores.get(model_col)
                row[model_col] = score
                native = native_map.get(model_col, 0.0)
                retention = (
                    (score / native * 100.0)
                    if (score is not None and native > 1e-9)
                    else None
                )
                row[f"{model_col} (retention %)"] = retention

                if retention is not None:
                    if model_col in [MODEL_EMBGEMMA, MODEL_QWEN_EMB]:
                        ded_retentions.append(retention)
                    else:
                        base_retentions.append(retention)

            row["mean_dedicated_retention_pct"] = (
                sum(ded_retentions) / len(ded_retentions) if ded_retentions else None
            )
            row["mean_base_retention_pct"] = (
                sum(base_retentions) / len(base_retentions) if base_retentions else None
            )

            rows.append(row)

    result_df = pl.DataFrame(rows)
    save_table_csv(result_df, out_file, precision=precision)
    return result_df


def build_suite_3_compression(
    master_lf: pl.LazyFrame, tables_dir: Path
) -> dict[str, pl.DataFrame]:
    """Build all 15 granular compression tables across every dataset and metric."""
    comp_dir = tables_dir / "compression"
    comp_dir.mkdir(parents=True, exist_ok=True)
    master_df = master_lf.collect()
    generated: dict[str, pl.DataFrame] = {}

    # 1. Retrieval compression (9 tables)
    for d in IR_DATASETS:
        d_short = DATASET_SHORT[d]
        for metric_name, m_short in IR_METRIC_SHORT.items():
            out_file = comp_dir / f"{d_short}_compression_{m_short}.csv"
            df = build_single_compression_table(
                master_df,
                dataset=d,
                track="retrieval",
                metric_name=metric_name,
                out_file=out_file,
                precision=4,
            )
            generated[f"{d_short}_compression_{m_short}"] = df

    # 2. Similarity compression (6 tables)
    for d in STS_DATASETS:
        d_short = DATASET_SHORT[d]
        for metric_name, m_short in STS_METRIC_SHORT.items():
            out_file = comp_dir / f"{d_short}_compression_{m_short}.csv"
            df = build_single_compression_table(
                master_df,
                dataset=d,
                track="similarity",
                metric_name=metric_name,
                out_file=out_file,
                precision=2,
            )
            generated[f"{d_short}_compression_{m_short}"] = df

    return generated
