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


def build_table_3_for_track(
    master_lf: pl.LazyFrame,
    track: str,
    metric_name: str,
    datasets: list[str],
    out_path: Path,
    precision: int = 4,
) -> pl.DataFrame:
    """Build pooling ablation table computing exact delta between last-token and mean pooling."""
    # 1. Filter base model full-dimension runs on this track and metric
    filtered_lf = master_lf.filter(
        (pl.col("track") == track)
        & (pl.col("regime") == "full")
        & (pl.col("metric") == metric_name)
        & (pl.col("tier") == "base")
        & (pl.col("dataset").is_in(datasets))
    )

    # 2. Average across datasets to get macro-performance per method and model
    macro_df = (
        filtered_lf.group_by(["method", "model"])
        .agg(pl.col("value").mean().alias("score"))
        .collect()
    )

    # 3. Pivot models into columns
    pivoted = macro_df.pivot(
        on="model",
        index="method",
        values="score",
    )

    # 4. Compute explicit deltas
    with_deltas = pivoted.with_columns(
        (pl.col(MODEL_GEMMA_LAST) - pl.col(MODEL_GEMMA_MEAN)).alias("gemma_delta"),
        (pl.col(MODEL_QWEN_BASE_LAST) - pl.col(MODEL_QWEN_BASE_MEAN)).alias(
            "qwen_delta"
        ),
    ).with_columns(
        ((pl.col("gemma_delta") + pl.col("qwen_delta")) / 2.0).alias("mean_delta")
    )

    # 5. Sort canonically by full methods
    ordered = order_full_methods(with_deltas)

    # Select final columns in logical presentation order
    final_cols = [
        "method",
        MODEL_GEMMA_MEAN,
        MODEL_GEMMA_LAST,
        "gemma_delta",
        MODEL_QWEN_BASE_MEAN,
        MODEL_QWEN_BASE_LAST,
        "qwen_delta",
        "mean_delta",
    ]
    result_df = ordered.select([c for c in final_cols if c in ordered.columns])
    save_table_csv(result_df, out_path, precision=precision)
    return result_df


def build_table_3_ablation(
    master_lf: pl.LazyFrame, out_dir: Path
) -> dict[str, pl.DataFrame]:
    """Generate Table 3 suite across IR and STS tracks."""
    ir_df = build_table_3_for_track(
        master_lf,
        track="retrieval",
        metric_name="ndcg_at_10",
        datasets=IR_DATASETS,
        out_path=out_dir / "table_3_pooling_ablation_ir.csv",
        precision=4,
    )
    sts_df = build_table_3_for_track(
        master_lf,
        track="similarity",
        metric_name="spearman_rho",
        datasets=STS_DATASETS,
        out_path=out_dir / "table_3_pooling_ablation_sts.csv",
        precision=2,
    )
    return {"ir": ir_df, "sts": sts_df}
