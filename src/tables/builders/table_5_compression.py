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


def build_table_5_for_track(
    master_lf: pl.LazyFrame,
    track: str,
    metric_name: str,
    datasets: list[str],
    out_path: Path,
    precision: int = 4,
) -> pl.DataFrame:
    """Build dimensional compression ladder table comparing methods and retention across dimensions."""
    # 1. Extract native baseline scores per model across datasets
    native_df = (
        master_lf.filter(
            (pl.col("track") == track)
            & (pl.col("regime") == "full")
            & (pl.col("method") == "baseline")
            & (pl.col("metric") == metric_name)
            & (pl.col("dataset").is_in(datasets))
        )
        .group_by("model")
        .agg(pl.col("value").mean().alias("native_score"))
        .collect()
    )
    native_map = dict(zip(native_df["model"], native_df["native_score"]))

    # 2. Extract compression runs averaged across datasets
    comp_df = (
        master_lf.filter(
            (pl.col("track") == track)
            & (pl.col("regime") == "compression")
            & (pl.col("metric") == metric_name)
            & (pl.col("dataset").is_in(datasets))
        )
        .group_by(["k", "method", "model"])
        .agg(pl.col("value").mean().alias("score"))
        .collect()
    )

    rows: list[dict[str, Any]] = []

    for k in LADDER_K_ORDERED:
        for m in METHODS_COMP_ORDERED:
            row: dict[str, Any] = {
                "k": k,
                "method": m,
            }

            sub = comp_df.filter((pl.col("k") == k) & (pl.col("method") == m))
            scores = dict(zip(sub["model"], sub["score"]))

            # Fill model scores and retention
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
                ret_col_name = f"{model_col} (retention %)"
                row[ret_col_name] = retention

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
    save_table_csv(result_df, out_path, precision=precision)
    return result_df


def build_table_5_compression(
    master_lf: pl.LazyFrame, out_dir: Path
) -> dict[str, pl.DataFrame]:
    """Generate Table 5 suite across IR and STS tracks."""
    ir_df = build_table_5_for_track(
        master_lf,
        track="retrieval",
        metric_name="ndcg_at_10",
        datasets=IR_DATASETS,
        out_path=out_dir / "table_5a_compression_ir.csv",
        precision=4,
    )
    sts_df = build_table_5_for_track(
        master_lf,
        track="similarity",
        metric_name="spearman_rho",
        datasets=STS_DATASETS,
        out_path=out_dir / "table_5b_compression_sts.csv",
        precision=2,
    )
    return {"ir": ir_df, "sts": sts_df}
