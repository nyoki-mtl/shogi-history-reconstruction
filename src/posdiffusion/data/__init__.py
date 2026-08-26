"""WebDataset-based data loading utilities for shogi position diffusion.

Module roles:
- webdataset_builder.py: Creates WebDataset shards from KIF files (data preparation phase)
- webdataset_loader.py: Core WebDataset loading functionality with ShogiWebDataset class (low-level API)
- trajectory_dataset.py: High-level dataset API wrapping webdataset_loader (recommended for training)

Usage:
1. Data preparation: Use webdataset_builder to convert KIF files to WebDataset format
2. Training: Use trajectory_dataset's create_trajectory_dataloader() for PyTorch Lightning
3. Custom usage: Use webdataset_loader directly for fine-grained control
"""

from .trajectory_dataset import (
    create_trajectory_dataloader,
    create_trajectory_dataset,
    get_dataset_info,
)
from .webdataset_builder import WebDatasetBuilder, create_webdataset_from_kif
from .webdataset_loader import ShogiWebDataset, create_webdataset_dataloader, load_metadata

__all__ = [
    "ShogiWebDataset",
    "WebDatasetBuilder",
    "create_trajectory_dataloader",
    "create_trajectory_dataset",
    "create_webdataset_dataloader",
    "create_webdataset_from_kif",
    "get_dataset_info",
    "load_metadata",
]
