from pathlib import Path

import polars as pl

from src.tables.common import (
    add_tier_summary_columns,
    order_full_methods,
    save_table_csv,
)
from src.tables.schema import CANONICAL_MODEL_COLUMNS, IR_DATASETS

IR_METRIC_SHORT: dict[str, str] = {
    "ndcg_at_10": "ndcg",
    "recall_at_100": "recall",
    "mrr_at_10": "mrr",
}


def build_suite_1_retrieval(
    master_lf: pl.LazyFrame, tables_dir: Path
) -> dict[str, pl.DataFrame]:
    """Build the 9 granular, self-contained full-dimension retrieval tables."""
    retrieval_dir = tables_dir / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, pl.DataFrame] = {}

    for d in IR_DATASETS:
        d_lower = d.lower().replace("2018", "")
        if d == "FiQA2018":
            d_lower = "fiqa"

        for metric_name, m_short in IR_METRIC_SHORT.items():
            # Filter strictly for this dataset and metric
            sub_lf = master_lf.filter(
                (pl.col("track") == "retrieval")
                & (pl.col("dataset") == d)
                & (pl.col("regime") == "full")
                & (pl.col("metric") == metric_name)
            )

            pivoted = sub_lf.collect().pivot(
                on="model",
                index="method",
                values="value",
                aggregate_function="mean",
            )

            with_summaries = add_tier_summary_columns(pivoted)
            ordered = order_full_methods(with_summaries)

            avail_models = [c for c in CANONICAL_MODEL_COLUMNS if c in ordered.columns]
            avail_summaries = [
                c
                for c in ["mean_dedicated", "mean_base_mean", "mean_base_last"]
                if c in ordered.columns
            ]
            final_cols = ["method"] + avail_models + avail_summaries
            result_df = ordered.select(final_cols)

            out_file = retrieval_dir / f"{d_lower}_{m_short}.csv"
            save_table_csv(result_df, out_file, precision=4)
            generated[f"{d_lower}_{m_short}"] = result_df

    return generated
