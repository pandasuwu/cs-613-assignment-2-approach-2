import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from kneed import KneeLocator

from src.config import EMBEDDINGS_CACHE_DIR
from src.tables.common import save_table_csv
from src.tables.schema import IR_DATASETS, STS_DATASETS
from src.transforms.common import compute_mean_and_cov
from src.transforms.compression.spectemp import calculate_snr_curve


def extract_spectemp_parameters_for_dataset(
    dataset: str,
    track: str,
    model_id: str = "google/embeddinggemma-300m",
) -> dict[str, Any]:
    """Compute noise floor variance, Kneedle knee point, and read derived gammas for a dataset."""
    clean_id = model_id.replace("/", "_")
    raw_dir = EMBEDDINGS_CACHE_DIR / dataset / "embedding" / clean_id / "raw"

    if track == "retrieval":
        calib_file = raw_dir / "corpus.pt"
        calib_data = torch.load(calib_file, weights_only=True)
    else:
        s1 = torch.load(raw_dir / "sentences1.pt", weights_only=True)
        s2 = torch.load(raw_dir / "sentences2.pt", weights_only=True)
        calib_data = torch.cat([s1, s2], dim=0)

    # Compute covariance and eigenvalues
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
    knee_point = kneedle.knee if kneedle.knee is not None else 1

    # Read derived gammas from cache
    gammas: dict[str, float] = {}
    for k in [512, 256, 128, 64]:
        gamma_file = (
            EMBEDDINGS_CACHE_DIR
            / dataset
            / "embedding"
            / clean_id
            / "compression"
            / f"spectemp_k{k}"
            / "gamma.json"
        )
        if gamma_file.exists():
            with open(gamma_file, encoding="utf-8") as f:
                gammas[f"gamma_k{k}"] = float(json.load(f)["gamma"])
        else:
            gammas[f"gamma_k{k}"] = 0.0

    return {
        "dataset": dataset,
        "track": track,
        "native_d": calib_data.shape[1],
        "noise_floor_variance": float(noise_var.item()),
        "kneedle_knee_point": int(knee_point),
        **gammas,
    }


def build_table_6_parameters(master_lf: pl.LazyFrame, out_dir: Path) -> pl.DataFrame:
    """Generate Table 6: SpecTemp adaptive tempering parameter dynamics."""
    rows: list[dict[str, Any]] = []

    for d in IR_DATASETS:
        rows.append(extract_spectemp_parameters_for_dataset(d, "retrieval"))

    for d in STS_DATASETS:
        rows.append(extract_spectemp_parameters_for_dataset(d, "similarity"))

    result_df = pl.DataFrame(rows)
    out_file = out_dir / "table_6_spectemp_dynamics.csv"
    save_table_csv(result_df, out_file, precision=4)
    return result_df
