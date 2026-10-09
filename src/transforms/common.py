import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project representation vectors onto the unit hypersphere S^{d-1}.

    Mathematical Formulation:
        x' = x / max(||x||_2, eps)

    Args:
        x: Input tensor of shape (N, d).
        eps: Lower bound to prevent division by zero for null vectors.

    Returns:
        Tensor of shape (N, d) where every row vector has unit L2 norm.
    """
    return F.normalize(x, p=2, dim=1, eps=eps)


def compute_mean_and_cov(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute empirical mean, descending eigenvalues, and orthogonal eigenvectors in FP32.

    Mathematical Formulation:
        mu = (1 / N) * sum_{i=1}^N x_i
        Sigma = (1 / (N - 1)) * sum_{i=1}^N (x_i - mu)(x_i - mu)^T
        Sigma = U * Lambda * U^T

    Since Sigma is a sample covariance matrix (Gram matrix), it is algebraically positive
    semi-definite (all eigenvalues lambda >= 0). Tiny negative eigenvalues (e.g. -1e-8)
    arising from FP32 discretization noise are projected to 0.0 onto the PSD cone.

    Args:
        x: Calibration tensor of shape (N, d).

    Returns:
        Tuple of:
            mu: Empirical mean vector of shape (d,).
            eigenvalues: Eigenvalues sorted in descending order of shape (d,).
            eigenvectors: Orthogonal matrix U of shape (d, d), where column j is the
                eigenvector corresponding to eigenvalues[j].
    """
    mu = torch.mean(x, dim=0)
    cov = torch.cov(x.T)
    # Perform symmetric eigendecomposition on CPU to avoid MPS threadgroup staging limits
    eigenvalues, eigenvectors = torch.linalg.eigh(cov.cpu())
    eigenvalues = eigenvalues.to(x.device)
    eigenvectors = eigenvectors.to(x.device)

    # Numerical projection onto the positive semi-definite cone
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    # Sort descending so component 0 represents the principal component of largest variance
    descending_idx = torch.argsort(eigenvalues, descending=True)
    return mu, eigenvalues[descending_idx], eigenvectors[:, descending_idx]
