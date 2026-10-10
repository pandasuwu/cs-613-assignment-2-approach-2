from pathlib import Path

import polars as pl

from src.tables.common import order_full_methods, save_table_csv
from src.tables.schema import (
    IR_DATASETS,
    MODEL_GEMMA_LAST,
    MODEL_GEMMA_MEAN,
    MODEL_QWEN_BASE_LAST,
    MODEL_QWEN_BASE_MEAN,
    STS_DATASETS,
)


def build_pooling_ablation_for_track(
    master_df: pl.DataFrame,
    track: str,
    metric_name: str,
    datasets: list[str],
    out_file: Path,
    precision: int = 4,
) -> pl.DataFrame:
    """Build pooling ablation table preserving full dataset-by-dataset granularity."""
    sub_df = master_df.filter(
        (pl.col("track") == track)
        & (pl.col("regime") == "full")
        & (pl.col("metric") == metric_name)
        & (pl.col("tier") == "base")
        & (pl.col("dataset").is_in(datasets))
    )

    pivoted = sub_df.pivot(
        on="model",
        index=["dataset", "method"],
        values="value",
    )

    with_deltas = pivoted.with_columns(
        (pl.col(MODEL_GEMMA_LAST) - pl.col(MODEL_GEMMA_MEAN)).alias("gemma_delta"),
        (pl.col(MODEL_QWEN_BASE_LAST) - pl.col(MODEL_QWEN_BASE_MEAN)).alias(
            "qwen_delta"
        ),
    ).with_columns(
        ((pl.col("gemma_delta") + pl.col("qwen_delta")) / 2.0).alias("mean_delta")
    )

    ordered_list: list[pl.DataFrame] = []
    for d in datasets:
        d_sub = with_deltas.filter(pl.col("dataset") == d)
        if len(d_sub) > 0:
            ordered_list.append(order_full_methods(d_sub))

    combined = pl.concat(ordered_list)

    final_cols = [
        "dataset",
        "method",
        MODEL_GEMMA_MEAN,
        MODEL_GEMMA_LAST,
        "gemma_delta",
        MODEL_QWEN_BASE_MEAN,
        MODEL_QWEN_BASE_LAST,
        "qwen_delta",
        "mean_delta",
    ]
    result_df = combined.select([c for c in final_cols if c in combined.columns])
    save_table_csv(result_df, out_file, precision=precision)
    return result_df


def build_suite_5_ablation(
    master_lf: pl.LazyFrame, tables_dir: Path
) -> dict[str, pl.DataFrame]:
    """Build the 2 pooling ablation tables for IR and STS."""
    ablation_dir = tables_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    master_df = master_lf.collect()

    ir_df = build_pooling_ablation_for_track(
        master_df,
        track="retrieval",
        metric_name="ndcg_at_10",
        datasets=IR_DATASETS,
        out_file=ablation_dir / "pooling_penalty_ir.csv",
        precision=4,
    )
    sts_df = build_pooling_ablation_for_track(
        master_df,
        track="similarity",
        metric_name="spearman_rho",
        datasets=STS_DATASETS,
        out_file=ablation_dir / "pooling_penalty_sts.csv",
        precision=2,
    )
    return {"ir": ir_df, "sts": sts_df}
