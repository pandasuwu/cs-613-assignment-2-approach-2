import numpy as np
import torch
from kneed import KneeLocator

from src.transforms.common import compute_mean_and_cov, l2_normalize


def calculate_snr_curve(
    eigenvalues: torch.Tensor, tail_percentile: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate local signal-to-noise ratio curve from eigenvalue spectrum (Li et al., 2026).

    Mathematical Formulation:
        tail_start = int(d * tail_percentile)
        sigma_noise^2 = mean(eigenvalues[tail_start:])
        SNR(i) = max(0, (lambda_i - sigma_noise^2) / sigma_noise^2)

    Assumes the trailing 10% of eigenvalues plateau into an empirical noise floor.

    Args:
        eigenvalues: Sorted descending eigenvalues of shape (d,).
        tail_percentile: Fraction of spectrum before noise tail (default: 0.9 for final 10%).

    Returns:
        Tuple of (snr_curve, noise_variance).
    """
    d = len(eigenvalues)
    tail_start = int(d * tail_percentile)
    noise_variance = torch.mean(eigenvalues[tail_start:])

    snr_curve = torch.clamp(
        (eigenvalues[:tail_start] - noise_variance)
        / torch.clamp(noise_variance, min=1e-12),
        min=0.0,
    )
    return snr_curve, noise_variance


def find_optimal_gamma(
    eigenvalues: torch.Tensor,
    target_dim: int,
    kneedle_s: float = 0.5,
    tail_percentile: float = 0.9,
) -> float:
    """Derive adaptive whitening exponent gamma(k) via Kneedle knee detection on SNR curve (Li et al., 2026).

    Mathematical Formulation:
        k_knee = Kneedle(SNR_curve, curve='convex', direction='decreasing', S=kneedle_s)
        gamma(k) = min(1.0, SNR(k) / SNR(k_knee))

    When target_dim <= k_knee, SNR(k) >= SNR(k_knee), yielding full whitening (gamma = 1.0).
    When target_dim > k_knee, gamma decays toward 0 (reverting toward PCA) as noise increases.

    Args:
        eigenvalues: Sorted descending eigenvalues of shape (d,).
        target_dim: Target dimension k.
        kneedle_s: Sensitivity parameter for Kneedle algorithm (default: 0.5).
        tail_percentile: Percentile cutoff for tail noise floor estimation.

    Returns:
        Adaptive tempering exponent gamma in [0.0, 1.0].
    """
    snr_curve, _ = calculate_snr_curve(eigenvalues, tail_percentile=tail_percentile)
    snr_np = snr_curve.cpu().numpy()
    n_points = len(snr_np)

    kneedle = KneeLocator(
        np.arange(1, n_points + 1),
        snr_np,
        curve="convex",
        direction="decreasing",
        S=kneedle_s,
    )

    knee_point = kneedle.knee
    if knee_point is None or knee_point < 1:
        knee_point = max(1, min(target_dim // 2, n_points))

    snr_knee = snr_np[knee_point - 1]
    if snr_knee <= 1e-12:
        return 1.0

    target_idx = min(target_dim, n_points) - 1
    snr_target = snr_np[target_idx]

    gamma = float(snr_target / snr_knee)
    return max(0.0, min(1.0, gamma))


def transform_spectemp(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    target_dim: int = 512,
    gamma: float | None = None,
    kneedle_s: float = 0.5,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Adaptive SNR-tempered spectral projection (SpecTemp, Li et al., 2026).

    Mathematical Formulation:
        mu, Lambda, U = cov(calib_data)
        gamma = find_optimal_gamma(Lambda, k) if gamma is None else gamma
        W_k = U_k @ diag((Lambda_k + epsilon)^(-gamma / 2))
        x' = (x - mu) @ W_k
        x' = x' / ||x'||_2

    Dynamically interpolates between unweighted PCA projection (gamma = 0) and
    full Whitening-k (gamma = 1) based on the signal-to-noise ratio of component k.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        calib_data: Calibration tensor of shape (K, d) used to estimate mean and covariance.
        target_dim: Target reduced dimension k (e.g. 512, 256, 128, 64).
        gamma: Optional fixed exponent. If None, automatically derived via Kneedle.
        kneedle_s: Kneedle sensitivity parameter.
        epsilon: Numerical stability parameter (default: 1e-6).

    Returns:
        Tuple of (transformed_x1, transformed_x2, gamma_value).
    """
    mu, eigenvalues, eigenvectors = compute_mean_and_cov(calib_data)

    if gamma is None:
        gamma_val = find_optimal_gamma(
            eigenvalues, target_dim=target_dim, kneedle_s=kneedle_s
        )
    else:
        gamma_val = float(gamma)

    u_k = eigenvectors[:, :target_dim]
    lam_k = eigenvalues[:target_dim]

    # Fractional eigenvalue scaling: (lambda_i + eps)^(-gamma / 2)
    scales = torch.clamp(lam_k + epsilon, min=1e-12) ** (-gamma_val / 2.0)
    w_k = u_k * scales.unsqueeze(0)

    x1_trans = torch.matmul(x1 - mu, w_k)
    x2_trans = torch.matmul(x2 - mu, w_k)

    return l2_normalize(x1_trans), l2_normalize(x2_trans), gamma_val
