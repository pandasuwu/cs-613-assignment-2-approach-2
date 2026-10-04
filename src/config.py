import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

# 1. Root and Directory Definitions
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
CACHE_DIR: Path = ROOT_DIR / "cache"
HF_HOME: Path = CACHE_DIR / "huggingface"
EMBEDDINGS_CACHE_DIR: Path = CACHE_DIR / "embeddings"
LOGS_DIR: Path = ROOT_DIR / "logs"
RESULTS_DIR: Path = ROOT_DIR / "results"

# 2. Enforce Project-Local Cache Isolation (Zero Home Directory Leakage)
os.environ["HF_HOME"] = str(HF_HOME)

# Propagate user Hugging Face authentication token if available without leaking cache
if "HF_TOKEN" not in os.environ:
    user_token_file = Path.home() / ".cache" / "huggingface" / "token"
    if user_token_file.exists():
        token = user_token_file.read_text().strip()
        os.environ["HF_TOKEN"] = token
        hf_token_dest = HF_HOME / "token"
        HF_HOME.mkdir(parents=True, exist_ok=True)
        hf_token_dest.write_text(token)

# Create mandatory directories
for directory in [
    CACHE_DIR,
    HF_HOME,
    EMBEDDINGS_CACHE_DIR,
    LOGS_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# 3. Scientific and Hardware Defaults
SEED: int = 2026
MAX_SEQ_LENGTH: int = 2048
DTYPE: torch.dtype = torch.float32

# Batch sizes optimized for Apple Silicon MPS (M3 Max 30-core GPU) and high-throughput execution
DEFAULT_BATCH_SIZE_DOCS: int = 64
DEFAULT_BATCH_SIZE_QUERIES: int = 128
DEFAULT_BATCH_SIZE_SEARCH: int = 512


def set_seed(seed: int = SEED) -> None:
    """Set deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Detect and return the highest-priority available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logger(name: str = "pipeline") -> logging.Logger:
    """Configure idiomatic logging to both stdout and project-local logs/ directory."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = LOGS_DIR / "experiment.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Initialize global random seed on import
set_seed(SEED)
logger: logging.Logger = setup_logger("pipeline")
