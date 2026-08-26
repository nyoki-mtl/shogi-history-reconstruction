"""PyTorch Lightning callbacks for training."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class EpochSetterCallback(pl.Callback):
    """Callback to update dataset's epoch at the beginning of each training epoch.

    This ensures that WebDataset uses different random seeds for each epoch
    when using persistent_workers=True.
    """

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the beginning of each training epoch.

        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The Lightning module being trained
        """
        # Access the train dataloader's dataset
        train_dataloader = trainer.train_dataloader
        if train_dataloader is not None and isinstance(train_dataloader, DataLoader):
            dataset = train_dataloader.dataset

            # Update epoch if the dataset supports it
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(trainer.current_epoch)
