import polars as pl

# 1. Canonical Polars Schema for Loading Atomic Results
RESULTS_SCHEMA: dict[str, type[pl.DataType]] = {
    "track": pl.Utf8,
    "dataset": pl.Utf8,
    "tier": pl.Utf8,
    "model": pl.Utf8,
    "pooling": pl.Utf8,
    "regime": pl.Utf8,
    "method": pl.Utf8,
    "k": pl.Int32,
    "gamma": pl.Float64,
    "metric": pl.Utf8,
    "value": pl.Float64,
}

# 2. Canonical Model Keys and Formatted Display Names
MODEL_EMBGEMMA: str = "google/embeddinggemma-300m"
MODEL_QWEN_EMB: str = "Qwen/Qwen3-Embedding-0.6B"
MODEL_GEMMA_MEAN: str = "google/gemma-3-1b-pt (mean_pooling)"
MODEL_GEMMA_LAST: str = "google/gemma-3-1b-pt (last_token_pooling)"
MODEL_QWEN_BASE_MEAN: str = "Qwen/Qwen3.5-0.8B-Base (mean_pooling)"
MODEL_QWEN_BASE_LAST: str = "Qwen/Qwen3.5-0.8B-Base (last_token_pooling)"

CANONICAL_MODEL_COLUMNS: list[str] = [
    MODEL_EMBGEMMA,
    MODEL_QWEN_EMB,
    MODEL_GEMMA_MEAN,
    MODEL_GEMMA_LAST,
    MODEL_QWEN_BASE_MEAN,
    MODEL_QWEN_BASE_LAST,
]

# Native dimensions per model
MODEL_NATIVE_DIMENSIONS: dict[str, int] = {
    MODEL_EMBGEMMA: 768,
    MODEL_QWEN_EMB: 1024,
    MODEL_GEMMA_MEAN: 1152,
    MODEL_GEMMA_LAST: 1152,
    MODEL_QWEN_BASE_MEAN: 1024,
    MODEL_QWEN_BASE_LAST: 1024,
}

# 3. Method Ordering
METHODS_FULL_ORDERED: list[str] = [
    "baseline",
    "standardization",
    "r1",
    "r2",
    "soft_zca",
    "abtt_1",
    "abtt_2",
    "abtt_3",
    "rand",
    "mc",
]

METHODS_COMP_ORDERED: list[str] = [
    "prefix",
    "pca",
    "whitening",
    "spectemp",
    "random_truncation",
]

LADDER_K_ORDERED: list[int] = [512, 256, 128, 64]

# 4. Datasets
IR_DATASETS: list[str] = ["FiQA2018", "ArguAna", "SCIDOCS"]
STS_DATASETS: list[str] = ["STSBenchmark", "SICK-R", "STS22.v2"]

DATASET_APPROX_TOKENS: dict[str, int] = {
    "SICK-R": 12,
    "STSBenchmark": 20,
    "FiQA2018": 75,
    "SCIDOCS": 200,
    "ArguAna": 250,
    "STS22.v2": 2048,
}
