import torch


def mean_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute mean pooling across active attention tokens."""
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    )
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def last_token_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Extract hidden state of the final non-padding token.

    When padding on the left, the final non-padding token is always at index -1.
    When padding on the right, the final non-padding token is at active length - 1.
    In benchmarks such as FiQA2018 (38 empty documents) and TRECCOVID (1 empty document),
    empty texts produce attention_mask.sum() == 0. Clamping to min=0 prevents PyTorch
    from wrapping negative indices to trailing padding tokens in right-padded batches.
    """
    if padding_side == "left":
        return hidden_states[:, -1, :]

    sequence_lengths = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
    batch_size = hidden_states.shape[0]
    return hidden_states[
        torch.arange(batch_size, device=hidden_states.device), sequence_lengths
    ]
