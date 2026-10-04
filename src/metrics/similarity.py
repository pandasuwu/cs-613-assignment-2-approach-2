import numpy as np
import scipy.stats
import torch


def compute_similarity_metrics(
    predictions: torch.Tensor | list[float] | np.ndarray,
    targets: torch.Tensor | list[float] | np.ndarray,
) -> dict[str, float]:
    """Compute Spearman rank correlation and Pearson linear correlation against human ratings.

    Mathematical Formulation:
        Spearman rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
        Pearson r = sum((p - mean(p)) * (t - mean(t))) / (sqrt(sum((p - mean(p))^2)) * sqrt(sum((t - mean(t))^2)))

    Args:
        predictions: Predicted sentence pair cosine similarities.
        targets: Human ground-truth semantic similarity ratings.

    Returns:
        Dictionary containing 'spearman_rho' and 'pearson_r', each scaled by 100 in [-100.0, 100.0].
    """
    p = (
        predictions.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor)
        else np.asarray(predictions)
    )
    t = (
        targets.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets)
    )

    spearman_res, _ = scipy.stats.spearmanr(p, t)
    pearson_res, _ = scipy.stats.pearsonr(p, t)

    spearman_val = 0.0 if np.isnan(spearman_res) else float(spearman_res * 100.0)
    pearson_val = 0.0 if np.isnan(pearson_res) else float(pearson_res * 100.0)

    return {"spearman_rho": spearman_val, "pearson_r": pearson_val}
