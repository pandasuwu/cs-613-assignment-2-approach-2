from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.stats

from src.tables.common import save_table_csv
from src.tables.schema import IR_DATASETS, STS_DATASETS


def compute_corr_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float]:
    """Compute Pearson r and Spearman rho alongside their two-sided p-values."""
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0, 0.0, 1.0
    r_val, p_r = scipy.stats.pearsonr(x, y)
    rho_val, p_rho = scipy.stats.spearmanr(x, y)
    return (
        0.0 if np.isnan(r_val) else float(r_val),
        1.0 if np.isnan(p_r) else float(p_r),
        0.0 if np.isnan(rho_val) else float(rho_val),
        1.0 if np.isnan(p_rho) else float(p_rho),
    )


def build_table_8_correlation(master_lf: pl.LazyFrame, out_dir: Path) -> pl.DataFrame:
    """Generate Table 8: Statistical correlation between intrinsic geometry and downstream task accuracy."""
    master_df = master_lf.collect()

    # 1. Prepare aligned IR dataset: geometry vs nDCG@10
    geom_ir = master_df.filter(
        (pl.col("track") == "geometry")
        & (pl.col("regime") == "full")
        & (pl.col("dataset").is_in(IR_DATASETS))
    )
    task_ir = master_df.filter(
        (pl.col("track") == "retrieval")
        & (pl.col("regime") == "full")
        & (pl.col("metric") == "ndcg_at_10")
        & (pl.col("dataset").is_in(IR_DATASETS))
    )

    # 2. Prepare aligned STS dataset: geometry vs Spearman rho
    geom_sts = master_df.filter(
        (pl.col("track") == "geometry")
        & (pl.col("regime") == "full")
        & (pl.col("dataset").is_in(STS_DATASETS))
    )
    task_sts = master_df.filter(
        (pl.col("track") == "similarity")
        & (pl.col("regime") == "full")
        & (pl.col("metric") == "spearman_rho")
        & (pl.col("dataset").is_in(STS_DATASETS))
    )

    geometry_metrics = [
        "centroid_norm",
        "average_cosine",
        "mev_top1",
        "nid_spectral_entropy",
        "isoscore",
    ]

    rows: list[dict[str, Any]] = []

    for gm in geometry_metrics:
        # Join IR
        g_sub_ir = geom_ir.filter(pl.col("metric") == gm)
        joined_ir = g_sub_ir.join(
            task_ir,
            on=["dataset", "model", "method"],
            suffix="_task",
        )
        x_ir = joined_ir["value"].to_numpy()
        y_ir = joined_ir["value_task"].to_numpy()
        r_ir, pr_ir, rho_ir, prho_ir = compute_corr_stats(x_ir, y_ir)

        # Join STS
        g_sub_sts = geom_sts.filter(pl.col("metric") == gm)
        joined_sts = g_sub_sts.join(
            task_sts,
            on=["dataset", "model", "method"],
            suffix="_task",
        )
        x_sts = joined_sts["value"].to_numpy()
        y_sts = joined_sts["value_task"].to_numpy()
        r_sts, pr_sts, rho_sts, prho_sts = compute_corr_stats(x_sts, y_sts)

        rows.append(
            {
                "geometry_metric": gm,
                "ir_ndcg_pearson_r": r_ir,
                "ir_ndcg_p_val": pr_ir,
                "ir_ndcg_spearman_rho": rho_ir,
                "sts_spearman_pearson_r": r_sts,
                "sts_spearman_p_val": pr_sts,
                "sts_spearman_spearman_rho": rho_sts,
            }
        )

    result_df = pl.DataFrame(rows)
    out_file = out_dir / "table_8_geometry_task_correlation.csv"
    save_table_csv(result_df, out_file, precision=4)
    return result_df
