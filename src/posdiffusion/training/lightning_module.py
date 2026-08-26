"""Optimized PyTorch Lightning module for Shogi position diffusion.

This module provides a high-performance implementation with:
- Fully automatic optimization (no manual backward)
- Integrated PCGrad optimizer
- Optional torch.compile support
- Clean, maintainable code structure
"""

import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from posdiffusion.board.constants import HAND_PIECE_OFFSET, MAX_PAWN_IN_HAND, TURN_BLACK, TURN_TOKEN_INDEX, TURN_WHITE
from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.board.validator import PositionValidator
from posdiffusion.config import Config
from posdiffusion.diffusion import (
    ConstrainedDiffusionLoss,
    DenoisingLoss,
    PCGradWrappedOptimizer,
    ShogiTransformerDenoiser,
)


class ShogiDiffusionLightningModule(pl.LightningModule):
    """Lightning module with manual optimization for PCGrad support."""

    def __init__(self, cfg: Config):
        super().__init__()
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(cfg, DictConfig):
                self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
            else:
                self.save_hyperparameters(cfg.__dict__)
        except ImportError:
            self.save_hyperparameters(cfg.__dict__)

        self.cfg = cfg

        # Model
        self.model = ShogiTransformerDenoiser(
            vocab_size=cfg.model.vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout,
            max_timesteps=cfg.model.num_timesteps,
            # Pass embedding usage down to model to avoid creating unused params
            use_game_progress=bool(getattr(cfg.model, "use_game_progress", False)),
        )

        # Loss function
        if hasattr(cfg, "loss") and cfg.loss.type == "constrained":
            self.loss_fn = ConstrainedDiffusionLoss(
                vocab_size=cfg.model.vocab_size,
                max_timesteps=cfg.model.num_timesteps,
                distance_stats=getattr(cfg.loss, "distance_stats", None),
                warmup_steps=getattr(cfg.loss, "warmup_steps", 10000),
                lambda_init=getattr(cfg.loss, "lambda_init", 0.1),
                lambda_max=getattr(cfg.loss, "lambda_max", 10.0),
                dual_lr=getattr(cfg.loss, "dual_lr", 1e-3),
                piece_dual_lr=getattr(cfg.loss, "piece_dual_lr", None),
                change_dual_lr=getattr(cfg.loss, "change_dual_lr", None),
                label_smoothing=getattr(cfg.loss, "label_smoothing", 0.0),
                use_piece_constraint=getattr(cfg.loss, "use_piece_constraint", True),
                # Change shaping (optional)
                change_concentration_weight=float(getattr(cfg.loss, "change_concentration_weight", 0.0) or 0.0),
                change_margin_weight=float(getattr(cfg.loss, "change_margin_weight", 0.0) or 0.0),
                change_margin_topk=int(getattr(cfg.loss, "change_margin_topk", 3) or 3),
                change_margin_m=float(getattr(cfg.loss, "change_margin_m", 0.5) or 0.5),
                change_count_weight=float(getattr(cfg.loss, "change_count_weight", 0.0) or 0.0),
                # Gumbel-Softmax alignment (optional)
                use_gumbel_change=bool(getattr(cfg.loss, "use_gumbel_change", False)),
                gumbel_tau=float(getattr(cfg.loss, "gumbel_tau", 1.0) or 1.0),
                gumbel_hard=bool(getattr(cfg.loss, "gumbel_hard", False)),
            )
            self.use_pcgrad = getattr(cfg.loss, "use_pcgrad", False)
            # Optional fine control: enable/disable change constraint or set initial lambdas
            if hasattr(self.loss_fn, "lambda_change"):
                if hasattr(cfg.loss, "use_change_constraint") and not cfg.loss.use_change_constraint:
                    # Disable change constraint by setting lambda to 0
                    with torch.no_grad():
                        self.loss_fn.lambda_change.data.zero_()
                piece_init = getattr(cfg.loss, "lambda_piece_init", None)
                if piece_init is not None:
                    with torch.no_grad():
                        self.loss_fn.lambda_piece.data.fill_(float(piece_init))
                change_init = getattr(cfg.loss, "lambda_change_init", None)
                if change_init is not None:
                    with torch.no_grad():
                        self.loss_fn.lambda_change.data.fill_(float(change_init))
        else:
            self.loss_fn = DenoisingLoss(reduction="mean", label_smoothing=0.0)
            self.use_pcgrad = False

        # Use MANUAL optimization for PCGrad compatibility
        self.automatic_optimization = False
        self.token_mask = self._build_token_mask()
        self._returnable_tokenizer: ShogiTokenizer | None = None
        self._returnable_validator: PositionValidator | None = None

        self.returnable_enabled = bool(getattr(cfg.training, "returnable_check", False))
        self.returnable_loss_weight = float(getattr(cfg.training, "returnable_loss_weight", 0.5))
        self.returnable_sample_prob = float(getattr(cfg.training, "returnable_sample_prob", 1.0))
        self.returnable_use_light_checks = bool(getattr(cfg.training, "returnable_use_light_checks", True))

    def _build_token_mask(self) -> torch.Tensor:
        """Build position-specific vocab mask for training-time constraints."""
        vocab_size = int(self.cfg.model.vocab_size)
        mask = torch.zeros(96, vocab_size, dtype=torch.bool)

        # Board tokens: allow piece ids (0-30), excluding unused ids 15 and 16.
        board_allowed = [i for i in range(min(vocab_size, 31)) if i not in (15, 16)]
        if board_allowed:
            mask[:81, board_allowed] = True

        # Hand tokens: allow hand count tokens (31-39).
        hand_start = HAND_PIECE_OFFSET
        hand_end = HAND_PIECE_OFFSET + MAX_PAWN_IN_HAND + 1
        hand_allowed = list(range(hand_start, min(vocab_size, hand_end)))
        if hand_allowed:
            mask[81:95, hand_allowed] = True

        # Turn token: allow 0/1 only.
        turn_allowed = [t for t in (TURN_BLACK, TURN_WHITE) if t < vocab_size]
        if turn_allowed:
            mask[TURN_TOKEN_INDEX, turn_allowed] = True

        return mask

    def _apply_token_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply training-time token mask if enabled in config."""
        if not getattr(self.cfg.training, "apply_token_mask", False):
            return logits
        mask = self.token_mask.to(device=logits.device)
        mask_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(~mask.unsqueeze(0), mask_value)

    def _ensure_returnable_tools(self) -> None:
        if self._returnable_tokenizer is None:
            self._returnable_tokenizer = ShogiTokenizer(flip_to_black=False)
            self._returnable_validator = PositionValidator(self._returnable_tokenizer)

    def _compute_returnable_mask(
        self,
        predicted_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> torch.Tensor:
        from posdiffusion.returnable import returnable_mask

        pred_cpu = predicted_tokens.detach().cpu().numpy()
        src_cpu = source_tokens.detach().cpu().numpy()
        mask_np = returnable_mask(
            pred_cpu,
            src_cpu,
            use_light_checks=self.returnable_use_light_checks,
        )
        return torch.from_numpy(mask_np).to(device=predicted_tokens.device)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, timesteps, **kwargs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Training step with manual optimization."""
        opt_obj = self.optimizers()
        opt0 = opt_obj[0] if isinstance(opt_obj, list) else opt_obj

        target = batch["target_tokens"]
        source = batch["source_tokens"]
        timesteps = batch.get("distance", torch.ones(target.shape[0], device=self.device, dtype=torch.long))
        absolute_move = batch.get("absolute_move") if self.cfg.model.use_game_progress else None
        game_phase = batch.get("game_phase") if self.cfg.model.use_game_progress else None

        logits: torch.Tensor | None = None

        if self.use_pcgrad and isinstance(self.loss_fn, ConstrainedDiffusionLoss):

            def pcgrad_closure() -> tuple[torch.Tensor, list[tuple[torch.Tensor, str]]]:
                """Computes losses and returns them along with logits."""
                nonlocal logits
                logits_local = self.model(
                    source,
                    timesteps,
                    absolute_move=absolute_move,
                    game_phase=game_phase,
                )
                logits_local = self._apply_token_mask(logits_local)
                loss_batch = {
                    "source_tokens": source,
                    "target_tokens": target,
                    "timesteps": timesteps,
                    "distances": timesteps,
                }
                losses = self.loss_fn.compute_individual_losses(logits_local, loss_batch, self.global_step)

                log_dict = {
                    "train/L_denoise": losses[0].detach(),
                    "train/L_piece": losses[1].detach(),
                    "train/L_change": losses[2].detach(),
                    "train/lambda_piece": self.loss_fn.lambda_piece.detach(),
                    "train/lambda_change": self.loss_fn.lambda_change.detach(),
                }
                self.log_dict(log_dict, batch_size=target.shape[0])

                # Only backpropagate through active losses
                active_losses = [(losses[0], "denoise")]
                constraints_active = self.global_step >= self.loss_fn.warmup_steps
                piece_enabled = bool(getattr(self.cfg.loss, "use_piece_constraint", True))
                change_enabled = bool(getattr(self.cfg.loss, "use_change_constraint", True))
                if constraints_active and piece_enabled:
                    active_losses.append((losses[1], "piece"))
                if constraints_active and change_enabled:
                    active_losses.append((losses[2], "change"))

                logits = logits_local
                return logits_local, active_losses

            # 1. Calculate gradients and get back loss and logits
            # Use PCGradWrappedOptimizer directly
            total_loss, logits_from_pc = opt0.pc_backward(closure=pcgrad_closure, backward_fn=self.manual_backward)  # type: ignore[attr-defined]
            logits = logits_from_pc  # Update logits with the one from the closure
            self.log("train/loss", total_loss, batch_size=target.shape[0])

            # 2. Manually clip gradients
            # Lightning wraps optimizer; skip type-enforced clip if underlying not exposed
            if hasattr(opt0, "_optimizer"):
                self.clip_gradients(
                    opt0._optimizer,  # type: ignore[attr-defined]
                    gradient_clip_val=self.cfg.training.gradient_clip_val,
                    gradient_clip_algorithm=self.cfg.training.gradient_clip_algorithm,
                )

            # 3. Step the optimizer
            opt0.step()

            # Update dual variables using the logits from the closure
            with torch.no_grad():
                constraints_active = self.global_step >= self.loss_fn.warmup_steps
                if constraints_active:
                    probs = F.softmax(logits, dim=-1)  # type: ignore[arg-type]
                    _, v_piece = self.loss_fn._fused_piece_constraint(probs, float(timesteps.float().mean().item()))
                    _, v_change = self.loss_fn._fused_change_constraint(probs, source, timesteps)
                    # Respect ablation flags: freeze updates for disabled constraints
                    if not bool(getattr(self.cfg.loss, "use_piece_constraint", True)):
                        v_piece = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
                    if not bool(getattr(self.cfg.loss, "use_change_constraint", True)):
                        v_change = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
                    self.loss_fn.update_dual_variables(v_piece, v_change)

        else:
            # Standard manual optimization (works for both constrained and denoising losses)
            logits = self.model(
                source,
                timesteps,
                absolute_move=absolute_move,
                game_phase=game_phase,
            )
            logits = self._apply_token_mask(logits)
            opt0.zero_grad()

            if isinstance(self.loss_fn, ConstrainedDiffusionLoss):
                # Build batch dict and compute full constrained loss
                loss_batch = {
                    "source_tokens": source,
                    "target_tokens": target,
                    "timesteps": timesteps,
                    "distances": timesteps,
                }
                loss_output = self.loss_fn(logits, loss_batch, self.global_step)
                total_loss = loss_output.total
                self.manual_backward(total_loss)

                # Gradient clipping (if base optimizer is wrapped)
                if hasattr(opt0, "_optimizer"):
                    self.clip_gradients(
                        opt0._optimizer,  # type: ignore[attr-defined]
                        gradient_clip_val=self.cfg.training.gradient_clip_val,
                        gradient_clip_algorithm=self.cfg.training.gradient_clip_algorithm,
                    )

                opt0.step()
                self.log("train/loss", total_loss, batch_size=target.shape[0])

                # Log components when piece constraint path is active
                if self.loss_fn.use_piece_constraint:
                    self.log_dict(
                        {
                            "train/L_denoise": loss_output.components.get(
                                "L_denoise", torch.tensor(0.0, device=target.device)
                            ),
                            "train/L_piece": loss_output.components.get(
                                "L_piece", torch.tensor(0.0, device=target.device)
                            ),
                            "train/L_change": loss_output.components.get(
                                "L_change", torch.tensor(0.0, device=target.device)
                            ),
                            "train/lambda_piece": getattr(
                                self.loss_fn, "lambda_piece", torch.tensor(0.0, device=target.device)
                            ).detach(),
                            "train/lambda_change": getattr(
                                self.loss_fn, "lambda_change", torch.tensor(0.0, device=target.device)
                            ).detach(),
                        },
                        batch_size=target.shape[0],
                    )

                # Update dual variables only after warmup (respect flags via returned violations)
                constraints_active = self.global_step >= self.loss_fn.warmup_steps
                if constraints_active:
                    v_piece = loss_output.violations.get("v_piece") if loss_output.violations is not None else None  # type: ignore[union-attr]
                    v_change = loss_output.violations.get("v_change") if loss_output.violations is not None else None  # type: ignore[union-attr]
                    if v_piece is not None and v_change is not None:
                        # Respect ablation flags: freeze updates for disabled constraints
                        if not bool(getattr(self.cfg.loss, "use_piece_constraint", True)):
                            v_piece = torch.tensor(0.0, device=target.device, dtype=torch.float32)
                        if not bool(getattr(self.cfg.loss, "use_change_constraint", True)):
                            v_change = torch.tensor(0.0, device=target.device, dtype=torch.float32)
                        self.loss_fn.update_dual_variables(v_piece, v_change)
            else:
                # Simple denoising loss
                per_sample_loss = self.loss_fn(logits, target, return_per_sample=True)
                if self.returnable_enabled and self.returnable_sample_prob > 0:
                    sample_mask = timesteps == 1
                    if sample_mask.any():
                        if self.returnable_sample_prob < 1.0:
                            prob_mask = torch.rand(target.shape[0], device=target.device) < self.returnable_sample_prob
                            sample_mask = sample_mask & prob_mask
                    if sample_mask.any():
                        pred_tokens = logits.argmax(dim=-1)
                        returnable_mask = self._compute_returnable_mask(
                            pred_tokens[sample_mask],
                            source[sample_mask],
                        )
                        weights = torch.ones_like(per_sample_loss)
                        weights[sample_mask] = torch.where(
                            returnable_mask.to(device=weights.device),
                            torch.full_like(weights[sample_mask], self.returnable_loss_weight),
                            torch.ones_like(weights[sample_mask]),
                        )
                        loss = (per_sample_loss * weights).mean()
                        self.log(
                            "train/returnable_rate",
                            returnable_mask.float().mean(),
                            batch_size=sample_mask.sum(),
                        )
                    else:
                        loss = per_sample_loss.mean()
                else:
                    loss = per_sample_loss.mean()

                self.manual_backward(loss)

                if hasattr(opt0, "_optimizer"):
                    self.clip_gradients(
                        opt0._optimizer,  # type: ignore[attr-defined]
                        gradient_clip_val=self.cfg.training.gradient_clip_val,
                        gradient_clip_algorithm=self.cfg.training.gradient_clip_algorithm,
                    )

                opt0.step()
                self.log("train/loss", loss, batch_size=target.shape[0])

        # Manually step the scheduler
        self.lr_scheduler_instance.step()

        # Log per-distance losses, ensuring logits is not None
        if logits is not None:
            unique_distances = torch.unique(timesteps)
            for dist in unique_distances:
                dist_mask = timesteps == dist
                if dist_mask.any():
                    dist_loss = F.cross_entropy(
                        logits[dist_mask].reshape(-1, logits.size(-1)),
                        target[dist_mask].reshape(-1),
                        reduction="mean",
                    )
                    self.log(f"train/loss_dist_{dist.item()}", dist_loss, batch_size=dist_mask.sum())

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer: torch.optim.Optimizer

        if self.use_pcgrad:
            # Pull pcgrad-specific config if available
            pcgrad_cfg = getattr(self.cfg.loss, "pcgrad_config", {}) or {}
            optimizer = PCGradWrappedOptimizer(
                self.model.parameters(),
                base_optimizer_class=torch.optim.AdamW,
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
                betas=(self.cfg.training.beta1, self.cfg.training.beta2),
                eps=float(pcgrad_cfg.get("eps", 1e-8)),
                randomize_order=bool(pcgrad_cfg.get("randomize_order", True)),
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
                betas=(self.cfg.training.beta1, self.cfg.training.beta2),
                eps=1e-8,
            )

        scheduler_type = getattr(self.cfg.training, "scheduler_type", "cosine")

        scheduler: LambdaLR | CosineAnnealingLR
        if scheduler_type == "cosine_with_warmup":
            max_epochs = int(getattr(self.cfg.training, "max_epochs", 1) or 1)
            limit_train_batches = int(getattr(self.cfg.training, "limit_train_batches", 1) or 1)
            num_training_steps = max_epochs * limit_train_batches
            num_warmup_steps = int(getattr(self.cfg.training, "warmup_steps", 0) or 0)

            def lr_lambda(current_step: int) -> float:
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                min_lr = float(getattr(self.cfg.training, "min_lr", 1e-6))
                min_lr_ratio = min_lr / float(self.cfg.training.learning_rate)
                return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

            scheduler = LambdaLR(optimizer, lr_lambda)

        elif scheduler_type == "cosine":
            max_epochs = int(getattr(self.cfg.training, "max_epochs", 1) or 1)
            limit_train_batches = int(getattr(self.cfg.training, "limit_train_batches", 1) or 1)
            t_max = max_epochs * limit_train_batches
            eta_min = float(getattr(self.cfg.training, "min_lr", 1e-6))
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

        self.lr_scheduler_instance = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        target = batch["target_tokens"]
        source = batch["source_tokens"]
        timesteps = batch.get("distance", torch.ones(target.shape[0], device=target.device, dtype=torch.long))
        absolute_move = batch.get("absolute_move") if self.cfg.model.use_game_progress else None
        game_phase = batch.get("game_phase") if self.cfg.model.use_game_progress else None
        logits = self.model(
            source,
            timesteps,
            absolute_move=absolute_move,
            game_phase=game_phase,
        )

        # --- Loss Logging ---
        total_loss: torch.Tensor
        if isinstance(self.loss_fn, ConstrainedDiffusionLoss):
            loss_batch = {
                "source_tokens": source,
                "target_tokens": target,
                "timesteps": timesteps,
                "distances": timesteps,
            }
            loss_output = self.loss_fn(logits, loss_batch, self.global_step)
            total_loss = loss_output.total
            self.log("val/loss", total_loss, batch_size=target.shape[0], sync_dist=True)
            if self.loss_fn.use_piece_constraint:
                self.log_dict(
                    {
                        "val/L_denoise": loss_output.components["L_denoise"],
                        "val/L_piece": loss_output.components["L_piece"],
                        "val/L_change": loss_output.components["L_change"],
                    },
                    batch_size=target.shape[0],
                    sync_dist=True,
                )
        else:
            total_loss = self.loss_fn(logits, target)
            self.log("val/loss", total_loss, batch_size=target.shape[0], sync_dist=True)

        # --- Per-Distance Loss Logging ---
        unique_distances = torch.unique(timesteps)
        for dist in unique_distances:
            dist_mask = timesteps == dist
            if dist_mask.any():
                # Use the same denoising loss calculation for apples-to-apples comparison
                dist_loss = F.cross_entropy(
                    logits[dist_mask].reshape(-1, logits.size(-1)),
                    target[dist_mask].reshape(-1),
                    reduction="mean",
                )
                self.log(f"val/loss_dist_{dist.item()}", dist_loss, batch_size=dist_mask.sum(), sync_dist=True)
