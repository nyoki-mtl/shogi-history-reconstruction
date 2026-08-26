"""Loss functions for backward diffusion model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DenoisingLoss(nn.Module):
    """Basic denoising loss for discrete diffusion."""

    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """Initialize denoising loss.

        Args:
            reduction: Reduction method ("mean" or "sum")
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        predicted_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        return_per_sample: bool = False,
    ) -> torch.Tensor:
        """Compute denoising loss.

        Args:
            predicted_logits: Predicted logits (batch_size, seq_len, vocab_size)
            target_tokens: Target tokens (batch_size, seq_len)
            token_mask: Optional mask for valid tokens (batch_size, seq_len)
            return_per_sample: If True, return per-sample loss (batch_size,) instead of scalar

        Returns:
            Scalar loss value or per-sample losses
        """
        batch_size, seq_len, vocab_size = predicted_logits.shape

        # Reshape for cross entropy
        logits_flat = predicted_logits.view(-1, vocab_size)
        targets_flat = target_tokens.view(-1)

        # Compute cross entropy loss
        if self.label_smoothing > 0:
            # Create smoothed targets
            smoothed_targets = torch.zeros_like(logits_flat).scatter_(
                1, targets_flat.unsqueeze(1), 1.0 - self.label_smoothing
            )
            smoothed_targets += self.label_smoothing / vocab_size

            # Compute loss with smoothed targets
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -(smoothed_targets * log_probs).sum(dim=-1)
        else:
            # Standard cross entropy
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Reshape back
        loss = loss.view(batch_size, seq_len)

        # Apply mask if provided
        if token_mask is not None:
            loss = loss * token_mask

        # Reduce
        if return_per_sample:
            # Return per-sample loss
            if token_mask is not None:
                # Average over valid tokens for each sample
                return loss.sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
            else:
                return loss.mean(dim=1)
        elif self.reduction == "mean":
            if token_mask is not None:
                # Average over valid tokens only
                return loss.sum() / token_mask.sum().clamp(min=1)
            else:
                return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
