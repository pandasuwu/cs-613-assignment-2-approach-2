from pathlib import Path

import polars as pl

from src.tables.common import save_table_csv
from src.tables.schema import IR_DATASETS, STS_DATASETS

DATASET_SHORT: dict[str, str] = {
    "FiQA2018": "fiqa",
    "ArguAna": "arguana",
    "SCIDOCS": "scidocs",
    "STSBenchmark": "stsb",
    "SICK-R": "sickr",
    "STS22.v2": "sts22",
}

GEOMETRY_COLS: list[str] = [
    "model",
    "method",
    "centroid_norm",
    "average_cosine",
    "mev_top1",
    "nid_spectral_entropy",
    "isoscore",
]


def build_suite_4_geometry(
    master_lf: pl.LazyFrame, tables_dir: Path
) -> dict[str, pl.DataFrame]:
    """Build the 6 granular geometry diagnostics tables (one per benchmark dataset)."""
    geom_dir = tables_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, pl.DataFrame] = {}

    all_datasets = IR_DATASETS + STS_DATASETS

    for d in all_datasets:
        d_short = DATASET_SHORT[d]

        sub_df = (
            master_lf.filter(
                (pl.col("track") == "geometry")
                & (pl.col("dataset") == d)
                & (
                    (pl.col("regime") == "full")
                    | ((pl.col("regime") == "compression") & (pl.col("k") == 128))
                )
            )
            .collect()
            .pivot(
                on="metric",
                index=["model", "method"],
                values="value",
            )
            .sort(["model", "method"])
        )

        final_cols = [c for c in GEOMETRY_COLS if c in sub_df.columns]
        result_df = sub_df.select(final_cols)

        out_file = geom_dir / f"{d_short}_geometry.csv"
        save_table_csv(result_df, out_file, precision=4)
        generated[f"{d_short}_geometry"] = result_df

    return generated
