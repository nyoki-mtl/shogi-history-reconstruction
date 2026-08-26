#!/usr/bin/env python3
"""Train the shogi position diffusion model from a YAML experiment config.

Usage:
    uv run scripts/train.py --config configs/experiments/baseline.yaml
    uv run scripts/train.py --config configs/experiments/constrained.yaml --resume
    uv run scripts/train.py --config configs/experiments/constrained.yaml training.fast_dev_run=true
"""

import argparse
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.model_summary import ModelSummary

from posdiffusion.config import Config, load_config
from posdiffusion.training.lightning_datamodule import create_datamodule
from posdiffusion.training.lightning_module import ShogiDiffusionLightningModule
from posdiffusion.utils.lightning_utils import (
    get_callbacks,
    get_device,
    get_logger,
    log_config,
    save_config,
    set_seed,
    update_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML (e.g. configs/experiments/baseline.yaml).",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Optional base config path (defaults to configs/base.yaml).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint of this experiment.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotlist overrides, e.g. training.batch_size=128 model.dropout=0.0",
    )
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the latest checkpoint in ``checkpoint_dir``, or None."""
    if not checkpoint_dir.exists():
        return None

    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if checkpoints:
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    return None


def resolve_run(cfg: Config, experiment_name: str, resume: bool) -> tuple[str, Path | None]:
    """Pick the run timestamp and optional resume checkpoint."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_base = Path(cfg.work_dir) / "experiments" / experiment_name

    if resume and experiment_base.exists():
        run_dirs = sorted(
            (d for d in experiment_base.iterdir() if d.is_dir()),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        for latest_run in run_dirs:
            ckpt_path = find_latest_checkpoint(latest_run / "checkpoints")
            if ckpt_path:
                print(f"📂 Resuming from checkpoint: {ckpt_path}")
                return latest_run.name, ckpt_path
        print(f"⚠️  No checkpoint found under {experiment_base}, starting fresh")

    return timestamp, None


def build_trainer(cfg: Config) -> pl.Trainer:
    return pl.Trainer(
        accelerator="auto",
        devices="auto" if cfg.training.devices == -1 else cfg.training.devices,
        strategy=cfg.training.strategy,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        limit_train_batches=cfg.training.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.training.get("limit_val_batches") or 1.0,
        logger=get_logger(cfg),
        callbacks=get_callbacks(cfg),
        log_every_n_steps=cfg.training.log_every_n_steps or 50,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        fast_dev_run=cfg.training.fast_dev_run,
        overfit_batches=cfg.training.overfit_batches,
        deterministic=cfg.training.deterministic,
        benchmark=cfg.training.benchmark,
        enable_checkpointing=True,
        enable_progress_bar=cfg.training.progress_bar["enabled"],
        enable_model_summary=True,
        reload_dataloaders_every_n_epochs=1,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, base=args.base, overrides=args.overrides)

    experiment_name = cfg.experiment.get("name") or args.config.stem
    timestamp, ckpt_path = resolve_run(cfg, experiment_name, args.resume)

    cfg.experiment_name = f"{experiment_name}/{timestamp}"

    # Logger naming and metadata (group runs of the same experiment together).
    if cfg.log.logger_type == "wandb":
        cfg.log.group = experiment_name
        cfg.log.run_name = f"{experiment_name}_{timestamp}"
        cfg.log.run_id = timestamp
        if cfg.experiment.get("tags"):
            cfg.log.tags = list(cfg.experiment.tags)
        if cfg.experiment.get("description"):
            cfg.log.notes = cfg.experiment.description

    print("🚀 Starting training")
    print(f"🎯 Experiment: {cfg.experiment_name}")
    print(f"📄 Config: {args.config}")

    cfg = update_paths(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    log_config(cfg)
    save_config(cfg, cfg.checkpoint_dir)
    print(f"💾 Config saved to: {cfg.checkpoint_dir}/config.yaml")
    print(f"🖥️  Using device: {device}")

    print("📊 Creating data module...")
    datamodule = create_datamodule(cfg)

    print("🧠 Creating model...")
    model = ShogiDiffusionLightningModule(cfg)

    trainer = build_trainer(cfg)

    if cfg.training.print_model_summary:
        print("\n" + "=" * 80)
        print("📋 Model Summary")
        print("=" * 80)
        print(ModelSummary(model, max_depth=2))
        print("=" * 80)

    datamodule.setup("fit")

    print("\n🎓 Starting training...")
    trainer.fit(model, datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)

    if cfg.training.save_final_model:
        final_model_path = Path(cfg.checkpoint_dir) / "final_model.ckpt"
        trainer.save_checkpoint(str(final_model_path))
        print(f"💾 Final model saved to: {final_model_path}")

    if cfg.training.run_test and hasattr(datamodule, "test_dataloader"):
        print("\n🧪 Running test evaluation...")
        print(f"📊 Test results: {trainer.test(model, datamodule)}")

    print(f"\n✅ Training complete. Artifacts in: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    # 'spawn' avoids deadlocks with multi-worker WebDataset DataLoaders.
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
