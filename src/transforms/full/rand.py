import torch

from src.transforms.common import l2_normalize


def transform_rand(
    x1: torch.Tensor,
    x2: torch.Tensor,
    seed: int = 2026,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random unit direction deflation negative control (Ren et al., 2025).

    Mathematical Formulation:
        v ~ N(0, I_d), u = v / ||v||_2
        x' = (x - (x . u) * u) / ||x - (x . u) * u||_2

    Functions as an empirical negative scientific control for R2. Deflating a single
    arbitrary random unit direction tests whether R2 improvements stem specifically
    from eliminating the true mean direction u_mu, or merely from reducing dimensionality
    by 1 subspace dimension.

    Args:
        x1: First representation tensor of shape (N, d) (documents in IR, sentences1 in STS).
        x2: Second representation tensor of shape (M, d) (queries in IR, sentences2 in STS).
        seed: Random seed for deterministic unit direction generation.

    Returns:
        Tuple of (transformed_x1, transformed_x2), each L2 unit-normalized.
    """
    # Initialize pseudo-random generator on CPU to guarantee bit-level identical random vectors
    # across heterogeneous hardware backends (Apple Silicon MPS, Linux CUDA, CPU).
    generator = torch.Generator(device="cpu").manual_seed(seed)
    d = x1.shape[1]
    v = torch.randn(d, generator=generator, device="cpu", dtype=x1.dtype).to(x1.device)
    u = v / torch.linalg.norm(v)

    x1_proj = (x1 @ u).unsqueeze(1) * u
    x2_proj = (x2 @ u).unsqueeze(1) * u

    x1_rand = x1 - x1_proj
    x2_rand = x2 - x2_proj

    return l2_normalize(x1_rand), l2_normalize(x2_rand)
