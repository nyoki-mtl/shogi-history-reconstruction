"""Utilities for PyTorch Lightning integration."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig as Config
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger


def get_logger(cfg: Config) -> Logger | bool:
    """Create a logger, or ``False`` to disable external logging.

    ``log.mode == "disabled"`` turns off logging; Lightning accepts
    ``logger=False``.
    """
    if cfg.log.mode == "disabled":
        return False

    if cfg.log.logger_type == "wandb":
        # Use custom group/name/id if provided in config
        group = getattr(cfg.log, "group", None)
        name = getattr(cfg.log, "run_name", cfg.experiment_name)
        run_id = getattr(cfg.log, "run_id", None)

        # Collect additional kwargs for wandb.init()
        # Collect tags and notes separately
        tags = list(cfg.log.tags) if hasattr(cfg.log, "tags") and cfg.log.tags is not None else None
        notes = cfg.log.notes if hasattr(cfg.log, "notes") and cfg.log.notes is not None else None

        return WandbLogger(
            project=cfg.log.project,
            entity=cfg.log.entity,
            name=name,
            group=group,
            id=run_id,
            save_dir=cfg.log.save_dir,
            offline=cfg.log.mode == "offline",
            log_model=cfg.log.log_model,
            tags=tags,
            notes=notes,
        )
    elif cfg.log.logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=cfg.log.save_dir,
            name=cfg.experiment_name,
            default_hp_metric=False,
        )
    else:
        raise ValueError(f"Unknown logger type: {cfg.log.logger_type}")


def get_callbacks(cfg: Config) -> list[Callback]:
    """Create callbacks based on configuration."""
    from pytorch_lightning.callbacks import (
        Callback,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )

    from posdiffusion.training.callbacks import EpochSetterCallback

    callbacks: list[Callback] = []

    # Add EpochSetterCallback for WebDataset epoch management
    callbacks.append(EpochSetterCallback())

    # Model checkpoint: keep the best `save_top_k` by val/loss, plus last.ckpt.
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename="epoch-{epoch:03d}_val-loss-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=cfg.training.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,  # Disable automatic metric name prefixes
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Progress bar with configurable formatting
    if cfg.training.progress_bar["enabled"]:

        class FixedFormatProgressBar(TQDMProgressBar):
            """Custom progress bar with fixed decimal places for metrics."""

            def get_metrics(self, trainer: Any, model: Any) -> dict[str, Any]:
                """Get metrics with fixed formatting."""
                items = super().get_metrics(trainer, model)

                # Format numeric values to fixed decimal places
                formatted_items: dict[str, Any] = {}
                for key, value in items.items():
                    if isinstance(value, float):
                        # Use 4 decimal places for loss values
                        if "loss" in key.lower():
                            formatted_items[key] = f"{value:.4f}"
                        # Use 2 decimal places for other metrics
                        else:
                            formatted_items[key] = f"{value:.2f}"
                    else:
                        formatted_items[key] = str(value)

                return formatted_items

        progress_bar = FixedFormatProgressBar()
        callbacks.append(progress_bar)

    return callbacks


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: Config) -> str:
    """Get device based on configuration and availability."""
    if cfg.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif cfg.device == "mps" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def save_config(cfg: Config, output_dir: str) -> None:
    """Save a fully-resolved snapshot of the configuration."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(output_dir) / "config.yaml"
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(resolved, config_path)


def load_checkpoint(
    checkpoint_path: str,
    map_location: str | None = None,
) -> dict[str, Any]:
    """Load checkpoint with proper device mapping."""
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint  # type: ignore[no-any-return]


def update_paths(cfg: Config) -> Config:
    """Update work_dir-derived paths based on the experiment name."""
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(exist_ok=True)

    experiment_dir = work_dir / "experiments" / cfg.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    cfg.checkpoint_dir = str(experiment_dir / "checkpoints")
    if hasattr(cfg, "log") and hasattr(cfg.log, "save_dir"):
        cfg.log.save_dir = str(experiment_dir / "logs")

    return cfg


def log_config(cfg: Config) -> None:
    """Print the configuration in a readable format."""
    print("\n" + "=" * 50)
    print("Configuration")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50 + "\n")
