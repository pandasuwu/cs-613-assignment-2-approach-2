from pathlib import Path
from typing import Any

import polars as pl

from src.tables.common import get_mean_scalar, save_table_csv
from src.tables.schema import IR_DATASETS, STS_DATASETS

DEFLATION_CANDIDATES: list[str] = ["r1", "r2", "abtt_1", "abtt_2", "abtt_3"]


def compute_gap_closure_rows(
    df: pl.DataFrame,
    datasets: list[str],
) -> list[dict[str, Any]]:
    """Compute gap closure statistics per dataset and for the macro-average."""
    rows: list[dict[str, Any]] = []

    all_d_eval = datasets + ["Macro-Average"]

    for d in all_d_eval:
        if d == "Macro-Average":
            sub_df = df.filter(pl.col("dataset").is_in(datasets))
        else:
            sub_df = df.filter(pl.col("dataset") == d)

        # 1. Dedicated baseline score
        ded_df = sub_df.filter(
            (pl.col("tier") == "embedding") & (pl.col("method") == "baseline")
        )
        ded_score = get_mean_scalar(ded_df, default=0.0)

        # 2. Raw base baseline score (mean pooling)
        base_raw_df = sub_df.filter(
            (pl.col("tier") == "base")
            & (pl.col("pooling") == "mean_pooling")
            & (pl.col("method") == "baseline")
        )
        base_raw_score = get_mean_scalar(base_raw_df, default=0.0)

        initial_gap = ded_score - base_raw_score

        # 3. Find best deflation method
        best_def_method = "r2"
        best_def_score = -1e9
        for m in DEFLATION_CANDIDATES:
            cand_df = sub_df.filter(
                (pl.col("tier") == "base")
                & (pl.col("pooling") == "mean_pooling")
                & (pl.col("method") == m)
            )
            score = get_mean_scalar(cand_df, default=-1e9)
            if score > best_def_score:
                best_def_score = score
                best_def_method = m

        # 4. Soft-ZCA score
        soft_df = sub_df.filter(
            (pl.col("tier") == "base")
            & (pl.col("pooling") == "mean_pooling")
            & (pl.col("method") == "soft_zca")
        )
        soft_score = get_mean_scalar(soft_df, default=0.0)

        def_closed_pct = (
            ((best_def_score - base_raw_score) / initial_gap * 100.0)
            if abs(initial_gap) > 1e-9
            else 0.0
        )
        soft_closed_pct = (
            ((soft_score - base_raw_score) / initial_gap * 100.0)
            if abs(initial_gap) > 1e-9
            else 0.0
        )

        rows.append(
            {
                "benchmark": d,
                "raw_base_score": base_raw_score,
                "dedicated_baseline": ded_score,
                "initial_gap": initial_gap,
                "best_deflation_method": best_def_method,
                "best_deflation_score": best_def_score,
                "deflation_gap_closed_pct": def_closed_pct,
                "soft_zca_score": soft_score,
                "soft_zca_gap_closed_pct": soft_closed_pct,
            }
        )

    return rows


def build_table_4_for_track(
    master_lf: pl.LazyFrame,
    track: str,
    metric_name: str,
    datasets: list[str],
    out_path: Path,
    precision: int = 4,
) -> pl.DataFrame:
    """Build Table 4 for a specific track and metric."""
    filtered_df = master_lf.filter(
        (pl.col("track") == track)
        & (pl.col("regime") == "full")
        & (pl.col("metric") == metric_name)
        & (pl.col("dataset").is_in(datasets))
    ).collect()

    rows = compute_gap_closure_rows(filtered_df, datasets)
    result_df = pl.DataFrame(rows)
    save_table_csv(result_df, out_path, precision=precision)
    return result_df


def build_table_4_gap_closure(
    master_lf: pl.LazyFrame, out_dir: Path
) -> dict[str, pl.DataFrame]:
    """Generate Table 4 suite across IR and STS tracks."""
    ir_df = build_table_4_for_track(
        master_lf,
        track="retrieval",
        metric_name="ndcg_at_10",
        datasets=IR_DATASETS,
        out_path=out_dir / "table_4_gap_closure_ir.csv",
        precision=4,
    )
    sts_df = build_table_4_for_track(
        master_lf,
        track="similarity",
        metric_name="spearman_rho",
        datasets=STS_DATASETS,
        out_path=out_dir / "table_4_gap_closure_sts.csv",
        precision=2,
    )
    return {"ir": ir_df, "sts": sts_df}
