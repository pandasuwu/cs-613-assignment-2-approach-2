import math

import torch

from src.transforms.common import l2_normalize


def compute_centroid_norm(x: torch.Tensor) -> float:
    """Compute the L2 norm of the empirical mean vector ||mu||_2 (Ren et al., 2025)."""
    return float(torch.linalg.norm(x.mean(dim=0)).item())


def compute_average_cosine(
    x: torch.Tensor, max_samples: int = 2000, seed: int = 2026
) -> float:
    """Compute average pairwise cosine similarity between random representations (Ethayarajh, 2019)."""
    n = x.shape[0]
    if n < 2:
        return 1.0

    normalized = l2_normalize(x)
    sample_size = min(n, max_samples)

    if sample_size < n:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(n, generator=generator)[:sample_size].to(x.device)
        sample = normalized[indices]
    else:
        sample = normalized

    gram = torch.matmul(sample, sample.T)
    off_diag_sum = torch.sum(gram) - torch.trace(gram)
    num_pairs = sample_size * (sample_size - 1)

    return float((off_diag_sum / num_pairs).item())


def compute_geometry_metrics(
    x: torch.Tensor, max_cosine_samples: int = 2000, seed: int = 2026
) -> dict[str, float]:
    """Compute all 5 intrinsic space geometry diagnostics in a single optimized pass.

    Metrics:
        - centroid_norm: Magnitude of empirical mean vector ||mu||_2 (Ren et al., 2025)
        - average_cosine: Mean pairwise cosine similarity across subsampled pairs (Ethayarajh, 2019)
        - mev_top1: Maximum Explainable Variance of leading eigenvalue lambda_1 / sum(lambda)
        - nid_spectral_entropy: Normalized Intrinsic Dimensionality H / log(d) (Yokoi et al., 2024)
        - isoscore: Covariance eigenspectrum defect from spherical identity (Rudman et al., 2022)

    Args:
        x: Representation tensor of shape (N, d).
        max_cosine_samples: Maximum vectors to sample for pairwise cosine computation.
        seed: Random seed for deterministic subsampling.

    Returns:
        Dictionary containing centroid_norm, average_cosine, mev_top1, nid_spectral_entropy, isoscore.
    """
    centroid_norm = compute_centroid_norm(x)
    average_cosine = compute_average_cosine(
        x, max_samples=max_cosine_samples, seed=seed
    )

    # Compute covariance eigenspectrum once for mev, nid, and isoscore on CPU
    eigenvalues = torch.clamp(torch.linalg.eigvalsh(torch.cov(x.T).cpu()), min=0.0).to(
        x.device
    )
    d = eigenvalues.shape[0]

    total_var = eigenvalues.sum()
    if total_var <= 1e-12:
        return {
            "centroid_norm": centroid_norm,
            "average_cosine": average_cosine,
            "mev_top1": 0.0,
            "nid_spectral_entropy": 0.0,
            "isoscore": 0.0,
        }

    # 1. Maximum Explainable Variance (MEV)
    mev = float((eigenvalues[-1] / total_var).item())

    # 2. Normalized Intrinsic Dimensionality (NID)
    probs = eigenvalues / total_var
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))
    max_entropy = math.log(d) if d > 1 else 1.0
    nid = float(torch.clamp(entropy / max_entropy, min=0.0, max=1.0).item())

    # 3. IsoScore (Rudman et al., 2022)
    norm = torch.linalg.norm(eigenvalues)
    cov_diag_normalized = (eigenvalues * math.sqrt(d)) / norm
    iso_diag = torch.ones(d, device=x.device, dtype=x.dtype)
    l2_distance = torch.linalg.norm(cov_diag_normalized - iso_diag)
    normalization_constant = math.sqrt(2.0 * (d - math.sqrt(d)))

    if normalization_constant <= 1e-12:
        isoscore = 1.0
    else:
        isotropy_defect = l2_distance / normalization_constant
        defect_term = (isotropy_defect**2) * (d - math.sqrt(d))
        raw_score = ((d - defect_term) ** 2 - d) / (d * (d - 1))
        isoscore = (
            float(torch.clamp(raw_score, min=0.0, max=1.0).item()) if d > 1 else 1.0
        )

    return {
        "centroid_norm": centroid_norm,
        "average_cosine": average_cosine,
        "mev_top1": mev,
        "nid_spectral_entropy": nid,
        "isoscore": isoscore,
    }


def compute_mev(x: torch.Tensor) -> float:
    """Compute Maximum Explainable Variance (MEV) of the leading principal component."""
    return compute_geometry_metrics(x)["mev_top1"]


def compute_nid(x: torch.Tensor) -> float:
    """Compute Normalized Intrinsic Dimensionality (NID) via normalized spectral entropy."""
    return compute_geometry_metrics(x)["nid_spectral_entropy"]


def compute_isoscore(points: torch.Tensor) -> float:
    """Compute IsoScore measuring covariance diagonal defect from identity (Rudman et al., 2022)."""
    return compute_geometry_metrics(points)["isoscore"]
