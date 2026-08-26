"""PyTorch Lightning DataModule for shogi position diffusion."""

from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.config import Config
from posdiffusion.data.trajectory_dataset import create_trajectory_dataloader
from posdiffusion.diffusion.schedules import DistanceSchedule


class ShogiDiffusionDataModule(pl.LightningDataModule):
    """Lightning DataModule for shogi diffusion training."""

    def __init__(self, cfg: Config):
        """Initialize the DataModule.

        Args:
            cfg: merged configuration (DictConfig)
        """
        super().__init__()
        self.cfg = cfg

        # Shared tokenizer instance
        self.tokenizer = ShogiTokenizer(flip_to_black=False)

        # Use model.num_timesteps if available, otherwise use distance config
        num_steps = cfg.model.num_timesteps

        self.distance_schedule = DistanceSchedule(
            num_steps=num_steps,
            distances=cfg.distance.schedule.distances,
            weights=cfg.distance.schedule.weights,
            curriculum_schedule=dict(cfg.distance.schedule.curriculum_schedule),
        )

        # Shared epoch counter for curriculum learning
        self.shared_epoch = torch.multiprocessing.Value("i", 0)

        # Data loaders
        self.train_dataloader_obj: DataLoader[Any] | None = None
        self.val_dataloader_obj: DataLoader[Any] | None = None
        self.test_dataloader_obj: DataLoader[Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for different stages."""
        # This method is called by Lightning before training, validation, or testing.
        # We create val/test dataloaders here to be reused.
        # The train dataloader is created on-the-fly in `train_dataloader()`.
        if (stage == "fit" or stage is None) and (self.val_dataloader_obj is None):
            self.val_dataloader_obj = create_trajectory_dataloader(
                shard_pattern=self.cfg.data.val_shard_pattern,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.data.val_num_workers,
                tokenizer=self.tokenizer,
                shuffle=False,
                augment=False,
                distance_schedule=self.distance_schedule,
                shared_epoch=self.shared_epoch,
                pin_memory=self.cfg.data.pin_memory,
                persistent_workers=self.cfg.data.persistent_workers,
                drop_last=False,
            )

        if (stage == "test" or stage is None) and self.test_dataloader_obj is None:
            self.test_dataloader_obj = create_trajectory_dataloader(
                shard_pattern=self.cfg.data.test_shard_pattern,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.data.val_num_workers,
                tokenizer=self.tokenizer,
                shuffle=False,
                augment=False,
                distance_schedule=self.distance_schedule,
                shared_epoch=self.shared_epoch,
                pin_memory=self.cfg.data.pin_memory,
                persistent_workers=self.cfg.data.persistent_workers,
                drop_last=False,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return training dataloader."""
        # With reload_dataloaders_every_n_epochs=1, this is called each epoch.
        # Update the epoch value just before creating the dataloader.
        if self.trainer:
            self.shared_epoch.value = self.trainer.current_epoch

        # Do not cache the dataloader object; a new one is created each epoch.
        if self.cfg.data.dataset_type == "webdataset":
            return create_trajectory_dataloader(
                shard_pattern=self.cfg.data.train_shard_pattern,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.data.train_num_workers,
                tokenizer=self.tokenizer,
                shuffle=True,
                augment=self.cfg.data.augment,
                distance_schedule=self.distance_schedule,
                shared_epoch=self.shared_epoch,
                shuffle_buffer_size=self.cfg.data.shuffle_buffer_size,
                pin_memory=self.cfg.data.pin_memory,
                persistent_workers=self.cfg.data.persistent_workers,
                repeat=True,
                drop_last=True,
            )
        raise ValueError(f"Unsupported dataset type: {self.cfg.data.dataset_type}")

    def val_dataloader(self) -> DataLoader[Any]:
        """Return validation dataloader."""
        if self.val_dataloader_obj is None:
            self.setup("fit")
        return self.val_dataloader_obj  # type: ignore[return-value]

    def test_dataloader(self) -> DataLoader[Any]:
        """Return test dataloader."""
        if self.test_dataloader_obj is None:
            self.setup("test")
        return self.test_dataloader_obj  # type: ignore[return-value]

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return prediction dataloader."""
        return self.test_dataloader()

    def teardown(self, stage: str | None = None) -> None:
        """Clean up after training."""
        # Clean up any resources if needed
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "cfg": self.cfg,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.cfg = state_dict["cfg"]


def create_datamodule(cfg: Config) -> ShogiDiffusionDataModule:
    """Create DataModule from configuration.

    Args:
        cfg: merged configuration (DictConfig)

    Returns:
        Configured DataModule
    """
    return ShogiDiffusionDataModule(cfg)
