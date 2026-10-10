from pathlib import Path
from typing import Any

import polars as pl

from src.config import RESULTS_DIR, logger
from src.tables.schema import RESULTS_SCHEMA

_CACHED_MASTER_DF: pl.DataFrame | None = None


def format_canonical_model_name(raw_model: str, tier: str, pooling: str | None) -> str:
    """Format raw directory model name into official Hugging Face identifier with pooling."""
    clean_model = raw_model.replace("google_", "google/").replace("Qwen_", "Qwen/")
    if tier == "base" and pooling:
        return f"{clean_model} ({pooling})"
    return clean_model


def load_master_dataframe(
    results_dir: Path = RESULTS_DIR, force_reload: bool = False
) -> pl.DataFrame:
    """Load all 2,160 atomic CSV leaves in results/ into a unified, strongly-typed Polars DataFrame."""
    global _CACHED_MASTER_DF
    if _CACHED_MASTER_DF is not None and not force_reload:
        return _CACHED_MASTER_DF

    logger.info("Scanning atomic results from %s", results_dir)
    records: list[dict[str, Any]] = []

    for p in results_dir.glob("**/*.csv"):
        parts = p.relative_to(results_dir).parts
        track = parts[0]
        dataset = parts[1]
        tier = parts[2]
        model_raw = parts[3]

        if tier == "base":
            pooling = parts[4]
            regime = parts[5]
            method_raw = p.stem
        else:
            pooling = None
            regime = parts[4]
            method_raw = p.stem

        canonical_model = format_canonical_model_name(model_raw, tier, pooling)

        df = pl.read_csv(p)
        if "metric" in df.columns and "value" in df.columns:
            for row in df.iter_rows(named=True):
                records.append(
                    {
                        "track": track,
                        "dataset": dataset,
                        "tier": tier,
                        "model": canonical_model,
                        "pooling": pooling,
                        "regime": regime,
                        "method": method_raw,
                        "k": None,
                        "gamma": None,
                        "metric": row["metric"],
                        "value": float(row["value"]),
                    }
                )
        elif "k" in df.columns:
            k_val = int(df["k"][0])
            gamma_val = (
                float(df["gamma"][0])
                if "gamma" in df.columns and df["gamma"][0] not in [None, ""]
                else None
            )
            method = method_raw.split("_k")[0]
            for col in df.columns:
                if col not in ["k", "gamma"]:
                    records.append(
                        {
                            "track": track,
                            "dataset": dataset,
                            "tier": tier,
                            "model": canonical_model,
                            "pooling": pooling,
                            "regime": regime,
                            "method": method,
                            "k": k_val,
                            "gamma": gamma_val,
                            "metric": col,
                            "value": float(df[col][0]),
                        }
                    )

    master_df = pl.DataFrame(records, schema=RESULTS_SCHEMA)
    logger.info(
        "Successfully loaded %d metric rows into master DataFrame", len(master_df)
    )
    _CACHED_MASTER_DF = master_df
    return master_df


def load_master_lazy(results_dir: Path = RESULTS_DIR) -> pl.LazyFrame:
    """Return a Polars LazyFrame wrapping the loaded master dataset for optimized query plans."""
    return load_master_dataframe(results_dir=results_dir).lazy()
