"""Constrained diffusion loss for piece-count and change-magnitude control.

The implementation uses vectorized constraint statistics and supports
``torch.compile``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from posdiffusion.board.constants import (
    BLACK,
    HAND_PIECE_INDICES,
    NONE,
    TARGET_PIECE_COUNTS,
    TURN_TOKEN_INDEX,
    TURN_WHITE,
    WHITE,
    generate_piece_mappings,
)


@dataclass
class ConstrainedDiffusionLossOutput:
    """Output structure for ConstrainedDiffusionLoss.

    Attributes:
        total: Total weighted loss for backward pass
        components: Individual loss components for logging/PCGrad
        violations: Constraint violations for dual variable updates
    """

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    violations: dict[str, torch.Tensor] | None = None


class ConstrainedDiffusionLoss(nn.Module):
    """Maximum performance constrained diffusion loss."""

    def __init__(
        self,
        vocab_size: int = 40,
        max_timesteps: int = 1000,
        distance_stats: dict[int, dict[str, float]] | None = None,
        warmup_steps: int = 10000,
        lambda_init: float = 0.1,
        lambda_max: float = 10.0,
        dual_lr: float = 1e-3,
        piece_dual_lr: float | None = None,
        change_dual_lr: float | None = None,
        label_smoothing: float = 0.0,
        use_piece_constraint: bool = True,
        # Change-shaping knobs (default off for backward-compat)
        change_concentration_weight: float = 0.0,
        change_margin_weight: float = 0.0,
        change_margin_topk: int = 3,
        change_margin_m: float = 0.5,
        # Change-count consistency shaping
        change_count_weight: float = 0.0,
        # Gumbel-Softmax option to align training with argmax inference
        use_gumbel_change: bool = False,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = False,
        # Side consistency (mover-side) shaping
        side_consistency_weight: float = 0.0,
        side_consistency_ratio_target: float = 0.2,
    ):
        super().__init__()

        # Require distance_stats to be provided
        if distance_stats is None:
            raise ValueError(
                "distance_stats must be provided. Please specify P5/P95 statistics for each distance used in training."
            )

        self.vocab_size = vocab_size
        self.max_timesteps = max_timesteps
        self.warmup_steps = warmup_steps
        self.lambda_max = lambda_max
        # Base and per-constraint dual learning rates (fallback to base)
        self.dual_lr = float(dual_lr)
        self.piece_dual_lr = float(piece_dual_lr) if piece_dual_lr is not None else float(dual_lr)
        self.change_dual_lr = float(change_dual_lr) if change_dual_lr is not None else float(dual_lr)
        self.label_smoothing = label_smoothing
        self.use_piece_constraint = use_piece_constraint
        # Change-shaping params
        self.change_concentration_weight = float(change_concentration_weight)
        self.change_margin_weight = float(change_margin_weight)
        self.change_margin_topk = int(change_margin_topk)
        self.change_margin_m = float(change_margin_m)
        # Change-count shaping
        self.change_count_weight = float(change_count_weight)
        # Gumbel-Softmax params
        self.use_gumbel_change = bool(use_gumbel_change)
        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_hard = bool(gumbel_hard)
        # Side consistency params
        self.side_consistency_weight = float(side_consistency_weight)
        self.side_consistency_ratio_target = float(side_consistency_ratio_target)

        # Target piece counts tensor - convert from dict to ordered list
        piece_order = ["pawn", "lance", "knight", "silver", "gold", "bishop", "rook"]
        target_counts_list = [float(TARGET_PIECE_COUNTS[p]) for p in piece_order]
        self.register_buffer("target_counts", torch.tensor(target_counts_list, dtype=torch.float32))

        # Lagrange multipliers are updated via dual ascent (no gradient descent)
        # Register as buffers to avoid DDP "unused parameters" issues
        self.register_buffer("lambda_piece", torch.tensor(lambda_init, dtype=torch.float32))
        self.register_buffer("lambda_change", torch.tensor(lambda_init, dtype=torch.float32))

        # Pre-compute all masks and statistics
        self._init_optimized_masks()
        self._init_optimized_stats(distance_stats)

    def _init_optimized_masks(self) -> None:
        """Initialize vectorized piece-count masks."""
        # Single consolidated mask tensor for all operations
        # Shape: [7, 96, 40] - piece_type, position, vocab
        masks = torch.zeros(7, 96, self.vocab_size, dtype=torch.float32)

        # Generate piece mappings from project constants
        piece_mappings = generate_piece_mappings()

        # Board positions (0-80) - fill masks from mapping
        for piece_id, piece_type in piece_mappings.items():
            masks[piece_type, :81, piece_id] = 1.0

        # Hand positions (81-94) - vectorized count weights
        hand_base = torch.arange(31, 40, dtype=torch.long)  # Count tokens
        hand_counts = torch.arange(0, 9, dtype=torch.float32)  # Actual counts

        # Use HAND_PIECE_INDICES for clarity
        for piece_type in range(7):
            black_pos = HAND_PIECE_INDICES[BLACK][piece_type]
            white_pos = HAND_PIECE_INDICES[WHITE][piece_type]
            masks[piece_type, black_pos, hand_base] = hand_counts
            masks[piece_type, white_pos, hand_base] = hand_counts

        self.register_buffer("piece_masks", masks)

        # Precompute token side/empty flags over vocab for board tokens
        token_is_black = torch.zeros(self.vocab_size, dtype=torch.float32)
        token_is_white = torch.zeros(self.vocab_size, dtype=torch.float32)
        token_is_empty = torch.zeros(self.vocab_size, dtype=torch.float32)

        # Mark NONE as empty if within vocab
        if 0 <= int(NONE) < self.vocab_size:
            token_is_empty[int(NONE)] = 1.0

        # Heuristic: piece id ranges in the token representation; black < 17, white >= 17.
        for pid in range(self.vocab_size):
            if token_is_empty[pid] == 1.0:
                continue
            if pid < 17:
                token_is_black[pid] = 1.0
            else:
                token_is_white[pid] = 1.0

        self.register_buffer("token_is_black", token_is_black)
        self.register_buffer("token_is_white", token_is_white)
        self.register_buffer("token_is_empty", token_is_empty)

    def _init_optimized_stats(self, distance_stats: dict[int, dict[str, float]]) -> None:
        """Pre-compute distance statistics as optimized tensors.

        Args:
            distance_stats: Dictionary mapping distance to P5/P95 statistics.
                           Required parameter (no fallback).
        """
        # Convert dict to tensors
        max_d = max(distance_stats.keys())
        p05 = torch.zeros(max_d + 1, dtype=torch.float32)
        p95 = torch.zeros(max_d + 1, dtype=torch.float32)

        for d, stats in distance_stats.items():
            p05[d] = stats["p05"]
            p95[d] = stats["p95"]

        # Store as contiguous tensors for optimal memory access
        self.register_buffer("dist_p05", p05.contiguous())
        self.register_buffer("dist_p95", p95.contiguous())
        self.max_dist = max_d

    def forward(
        self, logits: torch.Tensor, batch: dict[str, torch.Tensor], step: int = 0
    ) -> ConstrainedDiffusionLossOutput:
        """Compute the constrained loss.

        Returns ConstrainedDiffusionLossOutput with total loss, components, and violations.
        """
        # Softmax with memory-efficient computation
        probs = F.softmax(logits, dim=-1)

        # Extract batch data
        target = batch["target_tokens"]
        source = batch["source_tokens"]
        distances = batch["distances"]
        timesteps = batch["timesteps"]

        # Denoising loss - fused operations
        L_denoise = self._fused_denoising_loss(logits, target)

        # Early exit during warmup: constraints are disabled only during warmup
        # Note: whether each constraint contributes is controlled via lambdas
        # (e.g., lambda_piece==0 when piece constraint is disabled) and by the
        # training loop (which may skip dual updates). Do not gate on
        # use_piece_constraint here to allow "change-only" ablations.
        if step < self.warmup_steps:
            zero = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            return ConstrainedDiffusionLossOutput(
                total=L_denoise,
                components={
                    "L_denoise": L_denoise.detach(),
                    "L_piece": zero,
                    "L_change": zero,
                },
                violations={
                    "v_piece": zero,
                    "v_change": zero,
                },
            )

        # Constraints with fused operations
        avg_t = float(timesteps.float().mean().item())
        L_piece, v_piece = self._fused_piece_constraint(probs, avg_t)
        if self.use_gumbel_change:
            L_change, v_change = self._gumbel_change_constraint(logits, source, distances)
        else:
            L_change, v_change = self._fused_change_constraint(probs, source, distances)

        # Optional change shaping (folded into L_change)
        if self.change_concentration_weight > 0.0:
            L_change = L_change + self.change_concentration_weight * self._change_concentration_penalty(probs, source)
        if self.change_margin_weight > 0.0:
            L_change = L_change + self.change_margin_weight * self._topk_margin_loss(logits, source)
        if self.side_consistency_weight > 0.0:
            L_change = L_change + self.side_consistency_weight * self._side_consistency_penalty(probs, source)

        # Optional: change-count consistency shaping
        cc_weight = self.change_count_weight
        if cc_weight > 0.0:
            # Expected change count over first 95 tokens
            source_probs = probs.gather(2, source.unsqueeze(-1)).squeeze(-1)
            expected_changes = (1.0 - source_probs)[:, :95].sum(dim=1)
            # True change count from targets vs source
            true_changes = (target[:, :95] != source[:, :95]).to(expected_changes.dtype).sum(dim=1)

            # distance==1 では、target→source（出力→入力）の1手戻りの観点で
            # 「持ち駒総数が増えている＝捕獲があった」を判定し、理想変化数を 3/2 に固定
            with torch.no_grad():
                is_dist_one = distances == 1
                if is_dist_one.any():
                    src_hand = source[:, 81:95]
                    tgt_hand = target[:, 81:95]
                    src_counts = torch.clamp(src_hand - 31, min=0, max=8).to(expected_changes.dtype).sum(dim=1)
                    tgt_counts = torch.clamp(tgt_hand - 31, min=0, max=8).to(expected_changes.dtype).sum(dim=1)
                    captured = tgt_counts > src_counts  # 出力局面(target)の手駒が入力局面(source)より増
                    ideal_changes = torch.where(
                        captured,
                        torch.tensor(3.0, device=expected_changes.device),
                        torch.tensor(2.0, device=expected_changes.device),
                    )
                    final_true_changes = torch.where(is_dist_one, ideal_changes, true_changes)
                else:
                    final_true_changes = true_changes

            L_cc = F.smooth_l1_loss(expected_changes, final_true_changes, reduction="mean")
            L_change = L_change + cc_weight * L_cc

        # Total loss computation
        total = L_denoise + self.lambda_piece * L_piece + self.lambda_change * L_change

        return ConstrainedDiffusionLossOutput(
            total=total,
            components={
                "L_denoise": L_denoise.detach(),
                "L_piece": L_piece.detach(),
                "L_change": L_change.detach(),
                # Log change-count shaping if enabled
                **({"L_change_count": L_cc.detach()} if cc_weight > 0.0 else {}),
                "lambda_piece": self.lambda_piece.detach(),
                "lambda_change": self.lambda_change.detach(),
            },
            violations={
                "v_piece": v_piece.detach(),
                "v_change": v_change.detach(),
            },
        )

    def compute_individual_losses(
        self, logits: torch.Tensor, batch: dict[str, torch.Tensor], step: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute individual loss components for PCGrad.

        Returns:
            Tuple of (denoising_loss, piece_loss, change_loss)
        """
        # Softmax with memory-efficient computation
        probs = F.softmax(logits, dim=-1)

        # Extract batch data
        target = batch["target_tokens"]
        source = batch["source_tokens"]
        distances = batch["distances"]
        timesteps = batch["timesteps"]

        # Denoising loss
        L_denoise = self._fused_denoising_loss(logits, target)

        # Early exit during warmup only (see note above)
        if step < self.warmup_steps:
            zero = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            return L_denoise, zero, zero

        # Constraints
        avg_t = float(timesteps.float().mean().item())
        L_piece, _ = self._fused_piece_constraint(probs, avg_t)
        if self.use_gumbel_change:
            L_change, _ = self._gumbel_change_constraint(logits, source, distances)
        else:
            L_change, _ = self._fused_change_constraint(probs, source, distances)
        if self.change_concentration_weight > 0.0:
            L_change = L_change + self.change_concentration_weight * self._change_concentration_penalty(probs, source)
        if self.change_margin_weight > 0.0:
            L_change = L_change + self.change_margin_weight * self._topk_margin_loss(logits, source)
        if self.side_consistency_weight > 0.0:
            L_change = L_change + self.side_consistency_weight * self._side_consistency_penalty(probs, source)

        # Apply dual weights
        weighted_piece = self.lambda_piece * L_piece
        weighted_change = self.lambda_change * L_change

        return L_denoise, weighted_piece, weighted_change

    def _fused_denoising_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Fused denoising loss computation."""
        # Reshape with view for zero-copy
        _B, _S, V = logits.shape
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)

        if self.label_smoothing > 0:
            # Optimized label smoothing (matching vectorized implementation)
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits_flat)
                smooth_targets.fill_(self.label_smoothing / (V - 1))
                smooth_targets.scatter_(1, targets_flat.unsqueeze(1), 1.0 - self.label_smoothing)

            # Use KL divergence for smoothed targets
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = F.kl_div(log_probs, smooth_targets, reduction="none").sum(dim=-1)
        else:
            # Direct cross entropy (most efficient)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        return loss.mean()

    def _fused_piece_constraint(self, probs: torch.Tensor, avg_timestep: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused piece constraint computation.

        This constraint ensures that the total piece count matches standard shogi rules.
        Note: Token 39 represents "8+ pieces", causing undercounting in rare cases
        where one side holds 9-18 pawns. This is acceptable as such cases are
        extremely rare in real games.

        Args:
            probs: Predicted probabilities [batch, positions, vocab]
            avg_timestep: Average timestep for annealing

        Returns:
            Tuple of (loss, violation)
        """
        # Annealing weight - constraints are stronger near the end of denoising
        w = 1.0 - (avg_timestep / self.max_timesteps)

        # Single einsum for all piece counts
        # probs: [batch, positions(96), vocab(40)] - predicted probabilities
        # piece_masks: [7 piece types, positions(96), vocab(40)] - counting masks
        # counts: [batch, 7 piece types] - total count for each piece type
        # The einsum computes: counts[b,k] = sum over p,v of probs[b,p,v] * piece_masks[k,p,v]
        counts = torch.einsum("bpv,kpv->bk", probs, self.piece_masks)

        # Fixed target counts - no special case handling
        targets = self.target_counts.unsqueeze(0).expand_as(counts)

        # Fused violation computation using squared error
        diff = counts - targets
        loss = 0.5 * (diff * diff).sum(dim=1).mean() * w
        violation = diff.abs().mean()

        return loss, violation

    def _gumbel_change_constraint(
        self, logits: torch.Tensor, source: torch.Tensor, distances: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Change constraint using Gumbel-Softmax samples as argmax proxy.

        Uses a single Gumbel sample per forward for efficiency. If gumbel_hard=True,
        this becomes straight-through one-hot; otherwise it's a soft one-hot.
        """
        # Check for out-of-range distances
        if (distances > self.max_dist).any():
            invalid_distances = distances[distances > self.max_dist].unique().cpu().tolist()
            raise ValueError(
                f"Encountered distances {invalid_distances} exceeding max_dist={self.max_dist}. "
                f"Please provide distance_stats for all distances used in training."
            )

        # Gumbel-Softmax over vocab
        y = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=self.gumbel_hard, dim=-1)
        # Probability/indicator assigned to source token at each position
        y_same = y.gather(2, source.unsqueeze(-1)).squeeze(-1)
        # Approximate discrete change count (exclude turn token)
        changes = (1.0 - y_same)[:, :95].sum(dim=1)

        # Distance-based pinball loss
        d_clamp = distances.clamp(1, self.max_dist)
        p5_values = self.dist_p05[d_clamp]
        p95_values = self.dist_p95[d_clamp]

        lower = F.relu(p5_values - changes)
        upper = F.relu(changes - p95_values)
        loss = (0.05 * lower + 0.95 * upper).mean()
        violation = (lower + upper).mean()

        return loss, violation

    def _fused_change_constraint(
        self, probs: torch.Tensor, source: torch.Tensor, distances: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused change constraint computation."""
        # Check for out-of-range distances
        if (distances > self.max_dist).any():
            invalid_distances = distances[distances > self.max_dist].unique().cpu().tolist()
            raise ValueError(
                f"Encountered distances {invalid_distances} exceeding max_dist={self.max_dist}. "
                f"Please provide distance_stats for all distances used in training."
            )

        # Efficient source probability extraction
        _B, _S, _V = probs.shape
        source_probs = probs.gather(2, source.unsqueeze(-1)).squeeze(-1)

        # Change computation (exclude turn token)
        changes = (1 - source_probs)[:, :95].sum(dim=1)

        # Distance lookup with bounds checking already done
        d_clamp = distances.clamp(1, self.max_dist)
        p5_values = self.dist_p05[d_clamp]
        p95_values = self.dist_p95[d_clamp]

        # Fused pinball loss
        lower = F.relu(p5_values - changes)
        upper = F.relu(changes - p95_values)
        loss = (0.05 * lower + 0.95 * upper).mean()
        violation = (lower + upper).mean()

        return loss, violation

    def _change_concentration_penalty(self, probs: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Entropy penalty to concentrate where changes happen.

        Compute c = 1 - p_same over the first 95 tokens, then w = c / sum(c).
        Minimize entropy H(w) to avoid spreading small change mass across many positions.
        """
        # probs: [B, S, V]; source: [B, S]
        source_probs = probs.gather(2, source.unsqueeze(-1)).squeeze(-1)
        c = (1.0 - source_probs)[:, :95]
        c_sum = c.sum(dim=1, keepdim=True) + 1e-8
        w = c / c_sum
        entropy = -(w * (w + 1e-12).log()).sum(dim=1)
        return entropy.mean()

    def _topk_margin_loss(self, logits: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Hinge margin loss on top-k positions likely to change.

        For each sample, choose top-k positions by c = 1 - p_same (stop-grad selection),
        then encourage max(non-source logit) to exceed source logit by margin m.
        """
        B, _S, _V = logits.shape
        # Compute c from current probabilities (stop-gradient for selection)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            source_probs = probs.gather(2, source.unsqueeze(-1)).squeeze(-1)
            c = (1.0 - source_probs)[:, :95]
            k = max(1, min(self.change_margin_topk, c.size(1)))
            topk_idx = torch.topk(c, k=k, dim=1, largest=True, sorted=False).indices  # [B, k]

        # Gather logits at selected positions: shape [B, k, V]
        batch_indices = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, topk_idx.size(1))
        sel_logits = logits[batch_indices, topk_idx]  # [B, k, V]
        # Source token ids at selected positions: [B, k]
        sel_source_ids = source[batch_indices, topk_idx]

        # Source logits: [B, k]
        src_logits = sel_logits.gather(2, sel_source_ids.unsqueeze(-1)).squeeze(-1)

        # Top-2 over vocab at selected positions
        top2_vals, top2_idx = torch.topk(sel_logits, k=2, dim=-1)
        max_vals = top2_vals[..., 0]
        max_idx = top2_idx[..., 0]
        second_vals = top2_vals[..., 1]

        # Max non-source logit: if max corresponds to source, take second; else take max
        max_other = torch.where(max_idx == sel_source_ids, second_vals, max_vals)

        # Hinge margin: want max_other - src_logits >= m  => src_logits - max_other <= -m
        delta = src_logits - max_other
        loss = torch.relu(self.change_margin_m + delta).mean()
        return loss

    def _side_consistency_penalty(self, probs: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Encourage change mass to align with mover side (opposite of current turn).

        For current turn WHITE (input is gote), mover is BLACK (sente).
        Penalize the ratio of non-mover mass over mover mass exceeding a threshold.
        """
        _B, _S, _V = probs.shape
        device = probs.device
        eps = 1e-8

        # Determine current turn from source turn token
        turn_tokens = source[:, TURN_TOKEN_INDEX]
        current_is_white = (turn_tokens == TURN_WHITE).float()
        mover_is_black = 1.0 - current_is_white

        # Board slice (0..80)
        board_len = 81
        source_board = source[:, :board_len]

        # Change mass c = 1 - p_same on board
        source_probs = probs.gather(2, source.unsqueeze(-1)).squeeze(-1)
        c_board = (1.0 - source_probs)[:, :board_len]

        # Token side/empty flags
        token_is_black = self.token_is_black.to(device)
        token_is_white = self.token_is_white.to(device)
        token_is_empty = self.token_is_empty.to(device)

        src_is_black = token_is_black[source_board]
        src_is_white = token_is_white[source_board]
        src_is_empty = token_is_empty[source_board]

        # Destination likelihoods (prev board side distributions)
        p_black = torch.einsum("bsv,v->bs", probs[:, :board_len, :], token_is_black)
        p_white = torch.einsum("bsv,v->bs", probs[:, :board_len, :], token_is_white)

        mass_src_black = (src_is_black * c_board).sum(dim=1)
        mass_src_white = (src_is_white * c_board).sum(dim=1)
        mass_dest_black = (src_is_empty * p_black).sum(dim=1)
        mass_dest_white = (src_is_empty * p_white).sum(dim=1)

        mover_mass = mover_is_black * (mass_src_black + mass_dest_black) + (1.0 - mover_is_black) * (
            mass_src_white + mass_dest_white
        )
        non_mover_mass = mover_is_black * (mass_src_white + mass_dest_white) + (1.0 - mover_is_black) * (
            mass_src_black + mass_dest_black
        )

        ratio = non_mover_mass / (mover_mass + eps)
        loss = torch.relu(ratio - self.side_consistency_ratio_target).mean()
        return loss

    def update_dual_variables(self, v_piece: torch.Tensor, v_change: torch.Tensor) -> None:
        """Update dual variables (Lagrange multipliers) based on constraint violations.

        This method should be called during training to update the dual variables.
        It's separated from the forward pass to maintain torch.compile compatibility.

        Args:
            v_piece: Piece constraint violation
            v_change: Change constraint violation
        """
        with torch.no_grad():
            self.lambda_piece.data.add_(self.piece_dual_lr * v_piece).clamp_(0, self.lambda_max)
            self.lambda_change.data.add_(self.change_dual_lr * v_change).clamp_(0, self.lambda_max)


class PCGradWrappedOptimizer(Optimizer):
    """PCGrad optimizer wrapper with improved efficiency and generality.

    This optimizer wraps existing PyTorch optimizers and implements
    Projected Gradient Descent for conflicting gradients.
    It's designed to be more general and efficient:
    - No dependency on specific loss cache structures
    - Support for arbitrary number of tasks
    - Efficient gradient computation using backward() with retain_graph
    - Compatible with PyTorch's optimizer interface
    """

    def __init__(
        self,
        params: Any,
        base_optimizer_class: type[torch.optim.Optimizer] = torch.optim.AdamW,
        eps: float = 1e-8,
        randomize_order: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize PCGrad wrapped optimizer.

        Args:
            params: Model parameters or parameter groups
            base_optimizer_class: Base optimizer class (default: AdamW)
            eps: Small epsilon for numerical stability
            **kwargs: Additional arguments for base optimizer
        """
        # Convert params to list if needed
        if isinstance(params, torch.nn.Module):
            params = params.parameters()
        param_list = list(params)

        # Ensure optimizer-specific defaults are present
        defaults = kwargs.copy()  # Start with passed kwargs

        # Use optimizer name for robust checking against wrapped optimizer classes
        optimizer_name = getattr(base_optimizer_class, "__name__", "")

        if optimizer_name == "AdamW":
            defaults.setdefault("lr", 1e-3)
            defaults.setdefault("betas", (0.9, 0.999))
            defaults.setdefault("eps", 1e-8)
            defaults.setdefault("weight_decay", 0.01)
            defaults.setdefault("amsgrad", False)
            defaults.setdefault("maximize", False)
            # For newer PyTorch versions
            defaults.setdefault("foreach", None)
            defaults.setdefault("capturable", False)
            defaults.setdefault("differentiable", False)
            defaults.setdefault("fused", None)
        elif optimizer_name == "Adam":
            defaults.setdefault("lr", 1e-3)
            defaults.setdefault("betas", (0.9, 0.999))
            defaults.setdefault("eps", 1e-8)
            defaults.setdefault("weight_decay", 0)
            defaults.setdefault("amsgrad", False)
            defaults.setdefault("maximize", False)
            # For newer PyTorch versions
            defaults.setdefault("foreach", None)
            defaults.setdefault("capturable", False)
            defaults.setdefault("differentiable", False)
            defaults.setdefault("fused", None)
            defaults.setdefault("decoupled_weight_decay", False)
        else:
            # For other optimizers, we don't set defaults as we don't know them.
            # The user must provide all necessary args in the config.
            pass

        super().__init__(param_list, defaults)

        # Create base optimizer - pass parameters directly, not param_groups
        # This allows the base optimizer to create its own param_groups with correct defaults
        self.base_opt: torch.optim.Optimizer = base_optimizer_class(param_list, **defaults)
        self.eps = eps
        self.randomize_order = bool(randomize_order)

        # Manually initialize step count for LRScheduler compatibility
        self._step_count = 0

    def pc_backward(
        self,
        closure: Callable[[], tuple[torch.Tensor, list[tuple[torch.Tensor, str]]]],
        backward_fn: Callable[..., None],
    ) -> tuple[float | None, torch.Tensor]:
        """
        Performs the backward pass for each loss and projects the gradients.
        Does NOT perform the optimizer step.
        Returns the total loss and the logits for further use.
        """
        logits, task_losses = closure()

        if not isinstance(task_losses, list | tuple) or not task_losses:
            raise ValueError("PCGrad closure must return a non-empty list of (loss, name) tuples.")

        # --- PCGrad Logic ---
        task_gradients = []
        total_loss = torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)

        for i, (task_loss, _task_name) in enumerate(task_losses):
            # Zero gradients before each backward pass
            self.zero_grad(set_to_none=True)

            # Retain graph for all but the last task's backward pass
            retain_graph = i < len(task_losses) - 1
            # Use the provided backward function (e.g., Lightning's manual_backward)
            backward_fn(task_loss, retain_graph=retain_graph)

            total_loss += task_loss.detach()

            # Collect gradients
            with torch.no_grad():
                grads = []
                # Find a device from the available gradients
                device = next(
                    (p.grad.device for group in self.param_groups for p in group["params"] if p.grad is not None), None
                )
                if device is None:  # Fallback if no gradients were produced
                    device = self.param_groups[0]["params"][0].device

                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            grads.append(p.grad.data.clone().view(-1).to(device))
                        else:
                            grads.append(torch.zeros(p.numel(), device=device, dtype=p.dtype))

                if grads:
                    task_gradients.append(torch.cat(grads))

        # Apply PCGrad projection if we have multiple conflicting tasks
        if len(task_gradients) >= 2:
            projected_grad = self._apply_pcgrad(task_gradients)

            # Write back the projected gradients
            with torch.no_grad():
                idx = 0
                for group in self.param_groups:
                    for p in group["params"]:
                        numel = p.numel()
                        if p.grad is not None:  # Only write back to params that had grads
                            p.grad.data = projected_grad[idx : idx + numel].view_as(p.grad)
                        idx += numel

        return total_loss.item(), logits

    def step(self, closure: Any = None) -> Any:
        """
        Performs the optimizer step.
        Assumes that gradients have already been computed and projected (e.g., via pc_backward).
        The closure argument is ignored, as the backward pass is now separate.
        """
        result = self.base_opt.step()
        # Manually increment the wrapper's step counter to conform to PyTorch's LRScheduler expectations
        self._step_count += 1
        return result

    def _apply_pcgrad(self, task_gradients: list[torch.Tensor]) -> torch.Tensor:
        """Apply PCGrad projection to task gradients.

        Args:
            task_gradients: List of flattened gradients for each task

        Returns:
            Projected gradient vector
        """
        # Optionally randomize projection order per step
        grads = task_gradients
        if self.randomize_order and len(grads) > 1:
            # Use torch permutation for reproducibility with torch RNG
            device = grads[0].device
            order = torch.randperm(len(grads), device=device).tolist()
            grads = [grads[i] for i in order]

        # Start with first task gradient
        g_proj = grads[0].clone()

        # Project onto each subsequent task gradient
        for gi in grads[1:]:
            # Compute dot product
            dot = (g_proj * gi).sum()

            # Only project if gradients are conflicting (negative dot product)
            if dot < 0:
                # Project g_proj onto the orthogonal space of gi
                gi_norm_sq = (gi * gi).sum() + self.eps
                g_proj = g_proj - (dot / gi_norm_sq) * gi

        return g_proj

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients."""
        self.base_opt.zero_grad(set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state dict."""
        return self.base_opt.state_dict()  # type: ignore[no-any-return]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state dict."""
        self.base_opt.load_state_dict(state_dict)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Return parameter groups."""
        if hasattr(self, "base_opt"):
            return self.base_opt.param_groups  # type: ignore[no-any-return]
        else:
            # During initialization, use the internal _param_groups
            return self._param_groups if hasattr(self, "_param_groups") else []

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        """Set parameter groups."""
        if hasattr(self, "base_opt"):
            self.base_opt.param_groups = value
        else:
            # During initialization, store temporarily
            self._param_groups = value
