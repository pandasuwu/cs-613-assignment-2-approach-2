import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.stats
import torch
from kneed import KneeLocator

from src.config import EMBEDDINGS_CACHE_DIR
from src.tables.common import get_mean_scalar, save_table_csv
from src.tables.schema import (
    DATASET_APPROX_TOKENS,
    IR_DATASETS,
    STS_DATASETS,
)
from src.transforms.common import compute_mean_and_cov
from src.transforms.compression.spectemp import calculate_snr_curve

DEFLATION_CANDIDATES: list[str] = ["r1", "r2", "abtt_1", "abtt_2", "abtt_3"]


# -------------------------------------------------------------
# 1. Gap-Closure Analysis
# -------------------------------------------------------------
def build_gap_closure_table(
    master_df: pl.DataFrame,
    track: str,
    metric_name: str,
    datasets: list[str],
    out_file: Path,
    precision: int = 4,
) -> pl.DataFrame:
    """Compute base-to-dedicated gap closure percentages per dataset."""
    rows: list[dict[str, Any]] = []

    for d in datasets:
        sub_df = master_df.filter(
            (pl.col("track") == track)
            & (pl.col("dataset") == d)
            & (pl.col("regime") == "full")
            & (pl.col("metric") == metric_name)
        )

        ded_df = sub_df.filter(
            (pl.col("tier") == "embedding") & (pl.col("method") == "baseline")
        )
        ded_score = get_mean_scalar(ded_df, default=0.0)

        base_raw_df = sub_df.filter(
            (pl.col("tier") == "base")
            & (pl.col("pooling") == "mean_pooling")
            & (pl.col("method") == "baseline")
        )
        base_raw_score = get_mean_scalar(base_raw_df, default=0.0)
        initial_gap = ded_score - base_raw_score

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

    result_df = pl.DataFrame(rows)
    save_table_csv(result_df, out_file, precision=precision)
    return result_df


# -------------------------------------------------------------
# 2. SpecTemp Dynamics and Spectrum Properties (Li 2026)
# -------------------------------------------------------------
def build_spectemp_dynamics_table(out_file: Path) -> pl.DataFrame:
    """Tabulate spectral properties: leading eigenvalue, noise floor, knee point, and gammas."""
    clean_id = "google_embeddinggemma-300m"
    rows: list[dict[str, Any]] = []

    all_specs = [(d, "retrieval") for d in IR_DATASETS] + [
        (d, "similarity") for d in STS_DATASETS
    ]

    for d, track in all_specs:
        raw_dir = EMBEDDINGS_CACHE_DIR / d / "embedding" / clean_id / "raw"

        if track == "retrieval":
            calib_file = raw_dir / "corpus.pt"
            calib_data = torch.load(calib_file, weights_only=True)
        else:
            s1 = torch.load(raw_dir / "sentences1.pt", weights_only=True)
            s2 = torch.load(raw_dir / "sentences2.pt", weights_only=True)
            calib_data = torch.cat([s1, s2], dim=0)

        _, eigenvalues, _ = compute_mean_and_cov(calib_data)
        snr_curve, noise_var = calculate_snr_curve(eigenvalues, tail_percentile=0.9)

        snr_np = snr_curve.cpu().numpy()
        kneedle = KneeLocator(
            np.arange(1, len(snr_np) + 1),
            snr_np,
            curve="convex",
            direction="decreasing",
            S=0.5,
        )
        knee_point = int(kneedle.knee) if kneedle.knee is not None else 1
        ref_knee_snr = (
            float(snr_np[knee_point - 1]) if knee_point <= len(snr_np) else 0.0
        )

        gammas: dict[str, float] = {}
        for k in [512, 256, 128, 64]:
            gf = (
                EMBEDDINGS_CACHE_DIR
                / d
                / "embedding"
                / clean_id
                / "compression"
                / f"spectemp_k{k}"
                / "gamma.json"
            )
            if gf.exists():
                with open(gf, encoding="utf-8") as f:
                    gammas[f"gamma_k{k}"] = float(json.load(f)["gamma"])
            else:
                gammas[f"gamma_k{k}"] = 0.0

        rows.append(
            {
                "dataset": d,
                "track": track,
                "native_d": calib_data.shape[1],
                "leading_eigenvalue_lambda1": float(eigenvalues[0].item()),
                "noise_floor_variance": float(noise_var.item()),
                "snr_rank1": float(snr_np[0]),
                "kneedle_knee_point": knee_point,
                "reference_knee_snr": ref_knee_snr,
                **gammas,
            }
        )

    result_df = pl.DataFrame(rows)
    save_table_csv(result_df, out_file, precision=4)
    return result_df


# -------------------------------------------------------------
# 3. Geometry vs Task Correlation
# -------------------------------------------------------------
def build_geometry_task_correlation_table(
    master_df: pl.DataFrame, out_file: Path
) -> pl.DataFrame:
    """Compute Pearson and Spearman correlation between geometry metrics and downstream task accuracy."""
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

    def calc_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
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
        gir = geom_ir.filter(pl.col("metric") == gm).join(
            task_ir, on=["dataset", "model", "method"], suffix="_task"
        )
        r_ir, pr_ir, rho_ir, _ = calc_stats(
            gir["value"].to_numpy(), gir["value_task"].to_numpy()
        )

        # Join STS
        gsts = geom_sts.filter(pl.col("metric") == gm).join(
            task_sts, on=["dataset", "model", "method"], suffix="_task"
        )
        r_sts, pr_sts, rho_sts, _ = calc_stats(
            gsts["value"].to_numpy(), gsts["value_task"].to_numpy()
        )

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
    save_table_csv(result_df, out_file, precision=4)
    return result_df


# -------------------------------------------------------------
# 4. R2 vs R1 Head-to-Head Error Cancellation (Ren 2025 H1)
# -------------------------------------------------------------
def build_r2_vs_r1_table(master_df: pl.DataFrame, out_file: Path) -> pl.DataFrame:
    """Compare R2 vs R1 directly to test first-order parallel error cancellation."""
    r1_df = master_df.filter(pl.col("method") == "r1")
    r2_df = master_df.filter(pl.col("method") == "r2")

    joined = r1_df.join(
        r2_df,
        on=["track", "dataset", "model", "metric"],
        suffix="_r2",
    ).filter(
        ((pl.col("track") == "retrieval") & (pl.col("metric") == "ndcg_at_10"))
        | ((pl.col("track") == "similarity") & (pl.col("metric") == "spearman_rho"))
    )

    result = joined.select(
        [
            pl.col("dataset").alias("benchmark"),
            pl.col("track"),
            pl.col("metric"),
            pl.col("model"),
            pl.col("value").alias("score_r1"),
            pl.col("value_r2").alias("score_r2"),
            (pl.col("value_r2") - pl.col("value")).alias("delta_r2_minus_r1"),
            (pl.col("value_r2") > pl.col("value")).alias("r2_wins"),
        ]
    ).sort(["track", "benchmark", "model"])

    save_table_csv(result, out_file, precision=4)
    return result


# -------------------------------------------------------------
# 5. ABTT Monotonicity Progression (Mu 2018)
# -------------------------------------------------------------
def build_abtt_monotonicity_table(
    master_df: pl.DataFrame, out_file: Path
) -> pl.DataFrame:
    """Track D=0 -> 1 -> 2 -> 3 principal component deflation progression."""
    abtt_methods = ["baseline", "abtt_1", "abtt_2", "abtt_3"]
    sub = master_df.filter(
        pl.col("method").is_in(abtt_methods)
        & (
            ((pl.col("track") == "retrieval") & (pl.col("metric") == "ndcg_at_10"))
            | ((pl.col("track") == "similarity") & (pl.col("metric") == "spearman_rho"))
        )
    )

    pivoted = sub.pivot(
        on="method",
        index=["track", "dataset", "model"],
        values="value",
    )

    with_checks = pivoted.with_columns(
        (
            (pl.col("baseline") <= pl.col("abtt_1"))
            & (pl.col("abtt_1") <= pl.col("abtt_2"))
            & (pl.col("abtt_2") <= pl.col("abtt_3"))
        ).alias("monotonic_ascent"),
        (pl.col("abtt_3") - pl.col("baseline")).alias("total_gain_d3_vs_d0"),
    ).sort(["track", "dataset", "model"])

    save_table_csv(with_checks, out_file, precision=4)
    return with_checks


# -------------------------------------------------------------
# 6. Dual-Regime Compression Divergence (Li 2026 Theorem 1)
# -------------------------------------------------------------
def build_compression_divergence_table(
    master_df: pl.DataFrame, out_file: Path
) -> pl.DataFrame:
    """Contrast k=512 noise amplification against k=64 variance preservation."""
    sub = master_df.filter(
        (pl.col("regime") == "compression")
        & (pl.col("k").is_in([512, 64]))
        & (pl.col("method").is_in(["pca", "whitening", "spectemp"]))
        & (
            ((pl.col("track") == "retrieval") & (pl.col("metric") == "ndcg_at_10"))
            | ((pl.col("track") == "similarity") & (pl.col("metric") == "spearman_rho"))
        )
    )

    sub = sub.with_columns(
        (pl.col("method") + "_k" + pl.col("k").cast(pl.Utf8)).alias("method_k")
    )

    pivoted = sub.pivot(
        on="method_k",
        index=["track", "dataset", "model"],
        values="value",
    )

    with_deltas = pivoted.with_columns(
        (pl.col("spectemp_k512") - pl.col("whitening_k512")).alias(
            "broad_subspace_spectemp_advantage"
        ),
        (pl.col("spectemp_k64") - pl.col("pca_k64")).alias(
            "compact_subspace_spectemp_advantage"
        ),
    ).sort(["track", "dataset", "model"])

    save_table_csv(with_deltas, out_file, precision=4)
    return with_deltas


# -------------------------------------------------------------
# 7. Scientific Negative Controls Validation
# -------------------------------------------------------------
def build_negative_controls_table(
    master_df: pl.DataFrame, out_file: Path
) -> pl.DataFrame:
    """Validate counterfactuals: r2 vs rand, r1 vs mc, and pca vs random_truncation."""
    sub_full = master_df.filter(
        (pl.col("regime") == "full")
        & (pl.col("method").is_in(["r1", "r2", "rand", "mc"]))
        & (
            ((pl.col("track") == "retrieval") & (pl.col("metric") == "ndcg_at_10"))
            | ((pl.col("track") == "similarity") & (pl.col("metric") == "spearman_rho"))
        )
    ).pivot(
        on="method",
        index=["track", "dataset", "model"],
        values="value",
    )

    sub_comp = master_df.filter(
        (pl.col("regime") == "compression")
        & (pl.col("k") == 128)
        & (pl.col("method").is_in(["pca", "random_truncation"]))
        & (
            ((pl.col("track") == "retrieval") & (pl.col("metric") == "ndcg_at_10"))
            | ((pl.col("track") == "similarity") & (pl.col("metric") == "spearman_rho"))
        )
    ).pivot(
        on="method",
        index=["track", "dataset", "model"],
        values="value",
    )

    joined = sub_full.join(sub_comp, on=["track", "dataset", "model"])

    with_deltas = joined.with_columns(
        (pl.col("r2") - pl.col("rand")).alias("delta_r2_vs_rand"),
        (pl.col("r1") - pl.col("mc")).alias("delta_r1_vs_mc"),
        (pl.col("pca") - pl.col("random_truncation")).alias("delta_pca_vs_randtrunc"),
    ).sort(["track", "dataset", "model"])

    save_table_csv(with_deltas, out_file, precision=4)
    return with_deltas


# -------------------------------------------------------------
# 8. Sequence Length & Context Scale Impact
# -------------------------------------------------------------
def build_context_length_table(master_df: pl.DataFrame, out_file: Path) -> pl.DataFrame:
    """Dissect how text length impacts baseline pooling and post-processing recovery."""
    ordered_datasets = [
        ("SICK-R", "similarity", "spearman_rho"),
        ("STSBenchmark", "similarity", "spearman_rho"),
        ("FiQA2018", "retrieval", "ndcg_at_10"),
        ("SCIDOCS", "retrieval", "ndcg_at_10"),
        ("ArguAna", "retrieval", "ndcg_at_10"),
        ("STS22.v2", "similarity", "spearman_rho"),
    ]

    rows: list[dict[str, Any]] = []

    for d, track, metric in ordered_datasets:
        sub = master_df.filter(
            (pl.col("dataset") == d)
            & (pl.col("track") == track)
            & (pl.col("regime") == "full")
            & (pl.col("metric") == metric)
        )

        ded_base = get_mean_scalar(
            sub.filter(
                (pl.col("tier") == "embedding") & (pl.col("method") == "baseline")
            )
        )
        base_mean_base = get_mean_scalar(
            sub.filter(
                (pl.col("tier") == "base")
                & (pl.col("pooling") == "mean_pooling")
                & (pl.col("method") == "baseline")
            )
        )
        base_last_base = get_mean_scalar(
            sub.filter(
                (pl.col("tier") == "base")
                & (pl.col("pooling") == "last_token_pooling")
                & (pl.col("method") == "baseline")
            )
        )
        soft_zca = get_mean_scalar(
            sub.filter(
                (pl.col("tier") == "base")
                & (pl.col("pooling") == "mean_pooling")
                & (pl.col("method") == "soft_zca")
            )
        )

        pooling_penalty = base_last_base - base_mean_base
        penalty_pct = (
            (pooling_penalty / base_mean_base * 100.0)
            if abs(base_mean_base) > 1e-9
            else 0.0
        )

        rows.append(
            {
                "dataset": d,
                "track": track,
                "approx_mean_tokens": DATASET_APPROX_TOKENS[d],
                "dedicated_baseline": ded_base,
                "base_mean_baseline": base_mean_base,
                "base_last_baseline": base_last_base,
                "pooling_penalty": pooling_penalty,
                "pooling_penalty_pct": penalty_pct,
                "soft_zca_recovered_score": soft_zca,
                "recovery_gain": soft_zca - base_mean_base,
            }
        )

    result_df = pl.DataFrame(rows)
    save_table_csv(result_df, out_file, precision=4)
    return result_df


# -------------------------------------------------------------
# 9. Post-Hoc Spectral Compression vs Native MRL
# -------------------------------------------------------------
def build_spectemp_vs_mrl_table(
    master_df: pl.DataFrame, out_file: Path
) -> pl.DataFrame:
    """Compare training-free spectral compression against native MRL prefix slicing on dedicated models."""
    sub = master_df.filter(
        (pl.col("tier") == "embedding")
        & (pl.col("regime") == "compression")
        & (pl.col("method").is_in(["prefix", "pca", "spectemp"]))
        & (
            ((pl.col("track") == "retrieval") & (pl.col("metric") == "ndcg_at_10"))
            | ((pl.col("track") == "similarity") & (pl.col("metric") == "spearman_rho"))
        )
    )

    pivoted = sub.pivot(
        on="method",
        index=["track", "dataset", "model", "k"],
        values="value",
    )

    with_comp = pivoted.with_columns(
        (pl.col("spectemp") - pl.col("prefix")).alias("delta_spectemp_minus_mrl"),
        (pl.col("spectemp") >= pl.col("prefix")).alias("spectemp_beats_or_ties_mrl"),
    ).sort(["track", "dataset", "model", "k"])

    save_table_csv(with_comp, out_file, precision=4)
    return with_comp


# -------------------------------------------------------------
# 10. Tri-Paradigm Degeneracy Benchmark (Timkey vs Ren vs Diera)
# -------------------------------------------------------------
def build_tri_paradigm_benchmark_table(
    master_df: pl.DataFrame, out_file: Path
) -> pl.DataFrame:
    """Grand comparison between Coordinate (Timkey), Subspace (Ren), and Spectral (Diera) paradigms."""
    target_methods = ["baseline", "standardization", "r2", "soft_zca"]

    sub = master_df.filter(
        (pl.col("tier") == "base")
        & (pl.col("regime") == "full")
        & (pl.col("method").is_in(target_methods))
        & (
            ((pl.col("track") == "retrieval") & (pl.col("metric") == "ndcg_at_10"))
            | ((pl.col("track") == "similarity") & (pl.col("metric") == "spearman_rho"))
        )
    )

    pivoted = sub.pivot(
        on="method",
        index=["track", "dataset", "model"],
        values="value",
    )

    rows: list[dict[str, Any]] = []
    for r in pivoted.iter_rows(named=True):
        b = r["baseline"]
        s = r["standardization"]
        r2_val = r["r2"]
        zca = r["soft_zca"]

        best_m = "Soft-ZCA"
        best_val = zca
        if s > best_val and s > r2_val:
            best_m = "Standardization"
            best_val = s
        elif r2_val > best_val and r2_val > s:
            best_m = "R2"
            best_val = r2_val

        rows.append(
            {
                "track": r["track"],
                "dataset": r["dataset"],
                "model": r["model"],
                "score_baseline": b,
                "score_standardization": s,
                "score_r2": r2_val,
                "score_soft_zca": zca,
                "best_paradigm": best_m,
                "max_gain_over_baseline": best_val - b,
            }
        )

    result_df = pl.DataFrame(rows).sort(["track", "dataset", "model"])
    save_table_csv(result_df, out_file, precision=4)
    return result_df


# -------------------------------------------------------------
# Main Suite 6 Orchestrator
# -------------------------------------------------------------
def build_suite_6_analysis(
    master_lf: pl.LazyFrame, tables_dir: Path
) -> dict[str, pl.DataFrame]:
    """Build all 10 deep theoretical analysis tables."""
    analysis_dir = tables_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    master_df = master_lf.collect()
    generated: dict[str, pl.DataFrame] = {}

    generated["gap_closure_ir"] = build_gap_closure_table(
        master_df,
        track="retrieval",
        metric_name="ndcg_at_10",
        datasets=IR_DATASETS,
        out_file=analysis_dir / "gap_closure_ir.csv",
        precision=4,
    )

    generated["gap_closure_sts"] = build_gap_closure_table(
        master_df,
        track="similarity",
        metric_name="spearman_rho",
        datasets=STS_DATASETS,
        out_file=analysis_dir / "gap_closure_sts.csv",
        precision=2,
    )

    generated["spectemp_dynamics"] = build_spectemp_dynamics_table(
        out_file=analysis_dir / "spectemp_dynamics.csv"
    )

    generated["geometry_task_correlation"] = build_geometry_task_correlation_table(
        master_df,
        out_file=analysis_dir / "geometry_task_correlation.csv",
    )

    generated["r2_vs_r1_head_to_head"] = build_r2_vs_r1_table(
        master_df,
        out_file=analysis_dir / "r2_vs_r1_head_to_head.csv",
    )

    generated["abtt_monotonicity"] = build_abtt_monotonicity_table(
        master_df,
        out_file=analysis_dir / "abtt_monotonicity.csv",
    )

    generated["compression_divergence"] = build_compression_divergence_table(
        master_df,
        out_file=analysis_dir / "compression_divergence.csv",
    )

    generated["negative_controls_validation"] = build_negative_controls_table(
        master_df,
        out_file=analysis_dir / "negative_controls_validation.csv",
    )

    generated["context_length_impact"] = build_context_length_table(
        master_df,
        out_file=analysis_dir / "context_length_impact.csv",
    )

    generated["spectemp_vs_mrl"] = build_spectemp_vs_mrl_table(
        master_df,
        out_file=analysis_dir / "spectemp_vs_mrl.csv",
    )

    generated["tri_paradigm_benchmark"] = build_tri_paradigm_benchmark_table(
        master_df,
        out_file=analysis_dir / "tri_paradigm_benchmark.csv",
    )

    return generated
