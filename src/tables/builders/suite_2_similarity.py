from pathlib import Path

import polars as pl

from src.tables.common import (
    add_tier_summary_columns,
    order_full_methods,
    save_table_csv,
)
from src.tables.schema import CANONICAL_MODEL_COLUMNS, STS_DATASETS

STS_DATASET_SHORT: dict[str, str] = {
    "STSBenchmark": "stsb",
    "SICK-R": "sickr",
    "STS22.v2": "sts22",
}

STS_METRIC_SHORT: dict[str, str] = {
    "spearman_rho": "spearman",
    "pearson_r": "pearson",
}


def build_suite_2_similarity(
    master_lf: pl.LazyFrame, tables_dir: Path
) -> dict[str, pl.DataFrame]:
    """Build the 6 granular, self-contained full-dimension semantic similarity tables."""
    sim_dir = tables_dir / "similarity"
    sim_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, pl.DataFrame] = {}

    for d in STS_DATASETS:
        d_short = STS_DATASET_SHORT[d]

        for metric_name, m_short in STS_METRIC_SHORT.items():
            sub_lf = master_lf.filter(
                (pl.col("track") == "similarity")
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

            out_file = sim_dir / f"{d_short}_{m_short}.csv"
            save_table_csv(result_df, out_file, precision=2)
            generated[f"{d_short}_{m_short}"] = result_df

    return generated
