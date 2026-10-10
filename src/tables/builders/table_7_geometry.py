from pathlib import Path

import polars as pl

from src.tables.common import save_table_csv
from src.tables.schema import IR_DATASETS, METHODS_FULL_ORDERED, STS_DATASETS

TARGET_GEOMETRY_METHODS: list[str] = METHODS_FULL_ORDERED + [
    "pca",
    "whitening",
    "spectemp",
]


def build_table_7_for_track(
    master_lf: pl.LazyFrame,
    datasets: list[str],
    out_path: Path,
) -> pl.DataFrame:
    """Build geometry diagnostics table reporting all 5 metrics averaged across datasets in a track."""
    # Filter geometry records on target datasets
    geom_lf = master_lf.filter(
        (pl.col("track") == "geometry")
        & (pl.col("dataset").is_in(datasets))
        & (
            (pl.col("regime") == "full")
            | ((pl.col("regime") == "compression") & (pl.col("k") == 128))
        )
    )

    # Average across datasets to get macro-geometry per (model, method, metric)
    macro_df = (
        geom_lf.group_by(["model", "method", "metric"])
        .agg(pl.col("value").mean().alias("score"))
        .collect()
    )

    # Pivot metrics to columns
    pivoted = macro_df.pivot(
        on="metric",
        index=["model", "method"],
        values="score",
    )

    # Sort logically by model then method
    ordered = pivoted.sort(["model", "method"])

    # Ensure canonical column order
    geom_cols = [
        "model",
        "method",
        "centroid_norm",
        "average_cosine",
        "mev_top1",
        "nid_spectral_entropy",
        "isoscore",
    ]
    result_df = ordered.select([c for c in geom_cols if c in ordered.columns])
    save_table_csv(result_df, out_path, precision=4)
    return result_df


def build_table_7_geometry(
    master_lf: pl.LazyFrame, out_dir: Path
) -> dict[str, pl.DataFrame]:
    """Generate Table 7 suite across IR and STS manifold tracks."""
    ir_df = build_table_7_for_track(
        master_lf,
        datasets=IR_DATASETS,
        out_path=out_dir / "table_7_geometry_ir.csv",
    )
    sts_df = build_table_7_for_track(
        master_lf,
        datasets=STS_DATASETS,
        out_path=out_dir / "table_7_geometry_sts.csv",
    )
    return {"ir": ir_df, "sts": sts_df}
