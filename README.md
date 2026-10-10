# Representation Learning via Embedding Post-Processing

Research codebase for training-free, parameter-free geometric post-processing of neural text representations across dedicated embedding models and base language models on MTEB v2.

## Evaluated Models

Dedicated Embedding Models:
- google/embeddinggemma-300m (308M parameters, d=768)
- Qwen/Qwen3-Embedding-0.6B (600M parameters, d=1024)

Base Language Models (evaluated under mean and last-token pooling):
- google/gemma-3-1b-pt (1.0B parameters, d=1152)
- Qwen/Qwen3.5-0.8B-Base (0.8B parameters, d=1024)

## Benchmarks (MTEB v2 English)

Information Retrieval:
- FiQA2018, ArguAna, SCIDOCS (core evaluation matrix; TRECCOVID pre-cached)

Semantic Textual Similarity (STS):
- STSBenchmark, SICK-R, STS22.v2

## Post-Processing Methods

Full-Dimension Transforms (d to d):
- baseline: Raw unit-normalized representations
- standardization: Coordinate-wise z-score scaling (Timkey and van Schijndel, 2021)
- r1: Direct empirical mean subtraction (Ren et al., 2025)
- r2: Mean-subspace projection deflation (Ren et al., 2025)
- soft_zca: Regularized ZCA whitening (Diera et al., 2024)
- abtt_1, abtt_2, abtt_3: Centered PCA deflation of top 1, 2, or 3 components (Mu and Viswanath, 2018)
- rand: Random unit direction deflation negative control (Ren et al., 2025)
- mc: Unnormalized mean centering negative control (Ren et al., 2025)

Compression Transforms (d to k for k in 512, 256, 128, 64):
- prefix: Coordinate prefix slicing (Matryoshka baseline)
- random_truncation: Uniform random coordinate sampling negative control
- pca: Top-k principal component projection (Li et al., 2026)
- whitening: Truncated Whitening-k variance equalization (Su et al., 2021)
- spectemp: Adaptive SNR spectral tempering via Kneedle knee detection (Li et al., 2026)

## Metrics

- Retrieval: nDCG@10 (primary), Recall@100, MRR@10
- Similarity: Spearman rank correlation rho (primary), Pearson correlation r
- Intrinsic Geometry: centroid_norm, average_cosine, mev_top1, nid_spectral_entropy, isoscore

## Setup and Environment

Requires Python 3.14 and uv:

```bash
uv sync
```

### Hugging Face Authentication

Models like `google/gemma-3-1b-pt` require accepting user license agreements on Hugging Face. Supply your access token via:

```bash
export HF_TOKEN="your_hf_token_here"
```

If you previously logged in with `huggingface-cli login`, the pipeline non-destructively reads and inherits your token from `~/.cache/huggingface/token`.

### Localized Cache Isolation

All model checkpoints, tokenizers, and datasets are strictly isolated within the project-local `cache/huggingface` directory. User home directories are never polluted.

## Execution Pipeline

The execution operates in four decoupled, idempotent, and resumable stages.

### Stage 0: Download and Cache Assets

Downloads all 4 model checkpoints and 7 benchmark datasets into the isolated project cache (`cache/huggingface/`):

```bash
# Download and verify all 4 models and 7 datasets
uv run scripts/download.py

# Or download specific subsets
uv run scripts/download.py --models-only
uv run scripts/download.py --tasks-only
uv run scripts/download.py --model google/embeddinggemma-300m
uv run scripts/download.py --task FiQA2018
```

### Stage 1: Encode and Cache Raw Representations

Extracts representations once per model, task, and pooling mode into `cache/embeddings/`:

```bash
# Run all models and core tasks
uv run scripts/encode.py

# Run a single combination
uv run scripts/encode.py --model google/embeddinggemma-300m --task FiQA2018
uv run scripts/encode.py --model Qwen/Qwen3.5-0.8B-Base --task FiQA2018 --pooling mean_pooling
```

### Stage 2: Post-Processing Transformations

Applies all full and compression transformations to raw cached tensors:

```bash
# Transform all cached combinations
uv run scripts/transform.py

# Transform a single combination
uv run scripts/transform.py --model google/embeddinggemma-300m --task FiQA2018
```

### Stage 3: Evaluation and Metric Generation

Evaluates transformed representations and writes atomic CSV leaves into `results/`:

```bash
# Evaluate information retrieval
uv run scripts/eval_retrieval.py

# Evaluate semantic textual similarity
uv run scripts/eval_similarity.py

# Evaluate intrinsic space geometry diagnostics
uv run scripts/eval_geometry.py
```

Pass `--overwrite` to any script to recompute existing cached artifacts or results.

### Stage 4: Compile Research Tables

Parses all 2,160 atomic results into 49 granular, publication-ready CSV tables in `tables/` using Polars in under 2 seconds:

```bash
# Generate all 49 tables across the 6 suites
uv run scripts/generate_tables.py

# Or generate a specific suite (1..6)
uv run scripts/generate_tables.py --suite 1  # Full-dimension retrieval (9 tables)
uv run scripts/generate_tables.py --suite 2  # Full-dimension similarity (6 tables)
uv run scripts/generate_tables.py --suite 3  # Compression ladder across all metrics (15 tables)
uv run scripts/generate_tables.py --suite 4  # Intrinsic geometry diagnostics (6 tables)
uv run scripts/generate_tables.py --suite 5  # Token pooling ablations (2 tables)
uv run scripts/generate_tables.py --suite 6  # Theoretical analysis tables (11 tables)
```

## Code Quality

```bash
uv run ruff check .
uv run ty check
```
