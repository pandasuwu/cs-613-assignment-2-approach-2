from pathlib import Path

import polars as pl
import polars.selectors as cs

from src.config import logger
from src.tables.schema import (
    METHODS_FULL_ORDERED,
    MODEL_EMBGEMMA,
    MODEL_GEMMA_LAST,
    MODEL_GEMMA_MEAN,
    MODEL_QWEN_BASE_LAST,
    MODEL_QWEN_BASE_MEAN,
    MODEL_QWEN_EMB,
)


def get_mean_scalar(df: pl.DataFrame, col: str = "value", default: float = 0.0) -> float:
    """Extract scalar mean from a column, with type-safe fallback."""
    if len(df) == 0:
        return default
    val = df[col].mean()
    if isinstance(val, (int, float)):
        return float(val)
    return default


def add_tier_summary_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add separate, non-confounding summary averages for dedicated and base model tiers."""
    exprs: list[pl.Expr] = []

    # Dedicated embedding tier average
    if MODEL_EMBGEMMA in df.columns and MODEL_QWEN_EMB in df.columns:
        exprs.append(
            ((pl.col(MODEL_EMBGEMMA) + pl.col(MODEL_QWEN_EMB)) / 2.0).alias(
                "mean_dedicated"
            )
        )

    # Base models mean pooling average
    if MODEL_GEMMA_MEAN in df.columns and MODEL_QWEN_BASE_MEAN in df.columns:
        exprs.append(
            ((pl.col(MODEL_GEMMA_MEAN) + pl.col(MODEL_QWEN_BASE_MEAN)) / 2.0).alias(
                "mean_base_mean"
            )
        )

    # Base models last token pooling average
    if MODEL_GEMMA_LAST in df.columns and MODEL_QWEN_BASE_LAST in df.columns:
        exprs.append(
            ((pl.col(MODEL_GEMMA_LAST) + pl.col(MODEL_QWEN_BASE_LAST)) / 2.0).alias(
                "mean_base_last"
            )
        )

    if exprs:
        return df.with_columns(exprs)
    return df


def order_full_methods(df: pl.DataFrame, method_col: str = "method") -> pl.DataFrame:
    """Sort DataFrame rows according to the canonical full-dimension method sequence."""
    order_map = {m: i for i, m in enumerate(METHODS_FULL_ORDERED)}
    return (
        df.with_columns(
            pl.col(method_col)
            .replace_strict(order_map, default=999, return_dtype=pl.Int32)
            .alias("__order")
        )
        .sort("__order")
        .drop("__order")
    )


def save_table_csv(
    df: pl.DataFrame,
    out_path: Path,
    precision: int = 4,
) -> None:
    """Save Polars DataFrame as a strictly formatted CSV table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = df.with_columns(cs.numeric().round(precision))
    formatted.write_csv(out_path)
    logger.info(
        "Saved table -> %s (%d rows, %d cols)", out_path, len(df), len(df.columns)
    )
