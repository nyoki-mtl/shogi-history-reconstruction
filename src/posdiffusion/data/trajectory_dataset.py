"""WebDataset-based trajectory dataset for shogi position diffusion.

This module provides a simplified interface for WebDataset-based trajectory datasets.
"""

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, IterableDataset

from posdiffusion.board.tokenizer import ShogiTokenizer

from .webdataset_loader import ShogiWebDataset, create_webdataset_dataloader


def create_trajectory_dataset(
    shard_pattern: str,
    tokenizer: ShogiTokenizer | None = None,
    augment: bool = True,
    **kwargs: Any,
) -> IterableDataset[dict[str, Any]]:
    """Create a WebDataset-based trajectory dataset.

    Args:
        shard_pattern: Pattern for shard files (e.g., "shards/shard_{00000..00099}.tar")
        tokenizer: Shared tokenizer instance
        augment: Whether to apply data augmentation
        **kwargs: Additional arguments for ShogiWebDataset

    Returns:
        ShogiWebDataset instance
    """
    tokenizer = tokenizer or ShogiTokenizer(flip_to_black=False)

    return ShogiWebDataset(
        shard_pattern=shard_pattern,
        tokenizer=tokenizer,
        augment=augment,
        **kwargs,
    )


def create_trajectory_dataloader(
    shard_pattern: str,
    batch_size: int,
    num_workers: int = 4,
    tokenizer: ShogiTokenizer | None = None,
    shuffle: bool = True,
    augment: bool = True,
    distance_schedule: Any | None = None,  # DistanceSchedule instance
    shared_epoch: Any | None = None,
    shuffle_buffer_size: int = 1000,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    repeat: bool = False,
    drop_last: bool = True,
    **dataset_kwargs: Any,
) -> DataLoader[dict[str, Any]]:
    """Create a DataLoader for WebDataset trajectory datasets.

    Args:
        shard_pattern: Pattern for shard files
        batch_size: Batch size
        num_workers: Number of worker processes
        tokenizer: Shared tokenizer instance
        shuffle: Whether to shuffle the data
        augment: Whether to apply data augmentation
        distance_schedule: DistanceSchedule instance for curriculum learning
        shared_epoch: Multiprocessing-safe epoch counter
        shuffle_buffer_size: Size of the shuffle buffer for WebDataset
        pin_memory: Whether to use pinned memory
        persistent_workers: Whether to use persistent workers
        repeat: Whether to repeat the dataset indefinitely (useful for training)
        **dataset_kwargs: Additional arguments for ShogiWebDataset

    Returns:
        DataLoader instance
    """
    return create_webdataset_dataloader(
        shard_pattern=shard_pattern,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        augment=augment,
        tokenizer=tokenizer,
        distance_schedule=distance_schedule,
        shared_epoch=shared_epoch,
        shuffle_buffer_size=shuffle_buffer_size,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        repeat=repeat,
        drop_last=drop_last,
        **dataset_kwargs,
    )


def get_dataset_info(shard_pattern: str) -> dict[str, Any]:
    """Get information about a WebDataset.

    Args:
        shard_pattern: Pattern for shard files

    Returns:
        Dictionary with dataset information
    """
    info = {
        "dataset_type": "webdataset",
        "shard_pattern": shard_pattern,
    }

    # Try to load metadata
    # Extract directory from shard pattern
    shard_path = Path(shard_pattern.split("{")[0])  # Get path before brace expansion
    metadata_path = shard_path.parent / "metadata.json"

    if metadata_path.exists():
        import json

        with metadata_path.open() as f:
            metadata = json.load(f)
            info.update(metadata)

    return info
