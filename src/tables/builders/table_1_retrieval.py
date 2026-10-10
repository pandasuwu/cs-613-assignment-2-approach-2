from pathlib import Path

import polars as pl

from src.tables.common import (
    add_tier_summary_columns,
    order_full_methods,
    save_table_csv,
)
from src.tables.schema import CANONICAL_MODEL_COLUMNS, IR_DATASETS


def build_table_1_for_metric(
    master_lf: pl.LazyFrame,
    metric_name: str,
    out_dir: Path,
) -> pl.DataFrame:
    """Build unified Table 1 matrix for a specific retrieval metric across datasets and models."""
    # 1. Filter relevant records
    filtered_lf = master_lf.filter(
        (pl.col("track") == "retrieval")
        & (pl.col("regime") == "full")
        & (pl.col("metric") == metric_name)
        & (pl.col("dataset").is_in(IR_DATASETS))
    )

    # 2. Pivot so models become columns
    pivoted = filtered_lf.collect().pivot(
        on="model",
        index=["dataset", "method"],
        values="value",
        aggregate_function="mean",
    )

    # 3. Compute Macro-Average across the 3 IR benchmarks
    macro_avg = (
        pivoted.group_by("method")
        .agg(
            [
                pl.col(col).mean()
                for col in CANONICAL_MODEL_COLUMNS
                if col in pivoted.columns
            ]
        )
        .with_columns(pl.lit("Macro-Average").alias("dataset"))
        .select(pivoted.columns)
    )

    # 4. Concatenate per-dataset rows with macro-average rows
    combined = pl.concat([pivoted, macro_avg])

    # 5. Add tier summaries
    with_summaries = add_tier_summary_columns(combined)

    # 6. Sort canonically: dataset then method order
    ordered_list: list[pl.DataFrame] = []
    for d in IR_DATASETS + ["Macro-Average"]:
        sub_df = with_summaries.filter(pl.col("dataset") == d)
        if len(sub_df) > 0:
            ordered_list.append(order_full_methods(sub_df))

    result_df = pl.concat(ordered_list)

    # Reorder columns: dataset, method, models, summaries
    avail_models = [c for c in CANONICAL_MODEL_COLUMNS if c in result_df.columns]
    avail_summaries = [
        c
        for c in ["mean_dedicated", "mean_base_mean", "mean_base_last"]
        if c in result_df.columns
    ]
    final_cols = ["dataset", "method"] + avail_models + avail_summaries
    result_df = result_df.select(final_cols)

    suffix = metric_name.split("_")[0]  # ndcg, recall, mrr
    if "ndcg" in metric_name:
        suffix = "ndcg"
    out_file = out_dir / f"table_1_ir_{suffix}.csv"
    save_table_csv(result_df, out_file, precision=4)
    return result_df


def build_table_1_retrieval(
    master_lf: pl.LazyFrame, out_dir: Path
) -> dict[str, pl.DataFrame]:
    """Generate Table 1 suite across nDCG@10, Recall@100, and MRR@10."""
    metrics = ["ndcg_at_10", "recall_at_100", "mrr_at_10"]
    tables = {}
    for m in metrics:
        tables[m] = build_table_1_for_metric(master_lf, m, out_dir)
    return tables
