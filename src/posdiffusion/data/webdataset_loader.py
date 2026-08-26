"""WebDataset-based dataset for shogi position diffusion."""

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import webdataset as wds
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.diffusion.schedules import DistanceSchedule


def _identity_splitter(src: Any) -> Any:
    yield from src


class ShogiWebDataset(IterableDataset[dict[str, Any]]):
    """WebDataset-based dataset for backward trajectory diffusion."""

    def __init__(
        self,
        shard_pattern: str,
        distance_schedule: DistanceSchedule,
        tokenizer: ShogiTokenizer | None = None,
        shuffle: bool = True,
        augment: bool = True,
        flip_prob: float = 0.5,
        shared_epoch: Any | None = None,
        shuffle_buffer_size: int = 1000,
        enable_node_worker_split: bool = True,
        repeat: bool = False,
    ):
        """Initialize WebDataset with variable distance support.

        Args:
            shard_pattern: Pattern for shard files (e.g., "shards/shard_{00000..00099}.tar")
            distance_schedule: DistanceSchedule instance for curriculum learning
            tokenizer: Tokenizer instance
            shuffle: Whether to shuffle samples
            augment: Whether to apply augmentation
            flip_prob: Probability of horizontal flip
            shared_epoch: Multiprocessing-safe epoch counter for curriculum learning
            shuffle_buffer_size: Size of the shuffle buffer for WebDataset
        """
        # WebDataset interprets backslashes as escape characters, so normalize
        # Windows paths before passing them to its URL expansion layer.
        normalized_pattern = shard_pattern.replace("\\", "/")
        if len(normalized_pattern) >= 3 and normalized_pattern[1:3] == ":/":
            normalized_pattern = f"file:{normalized_pattern}"
        self.shard_pattern = normalized_pattern
        self.tokenizer = tokenizer or ShogiTokenizer(flip_to_black=False)
        self.shuffle = shuffle
        self.augment = augment
        self.flip_prob = flip_prob
        self.shared_epoch = shared_epoch
        self.distance_schedule = distance_schedule
        self.shuffle_buffer_size = shuffle_buffer_size
        self.enable_node_worker_split = enable_node_worker_split
        self.repeat = repeat

        # Pre-compute flip indices
        self._flip_indices = np.array(
            [9 * (8 - i) + j for i in range(9) for j in range(9)] + list(range(81, 96)),
            dtype=np.int32,
        )

    def _get_current_epoch(self) -> int:
        """Get the current epoch from the shared value."""
        return self.shared_epoch.value if self.shared_epoch is not None else 0

    def _get_current_weights(self) -> list[float]:
        """Get sampling weights based on current epoch for curriculum learning."""
        current_epoch = self._get_current_epoch()
        return self.distance_schedule.get_weights_for_epoch(current_epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples with epoch-aware random seeding and curriculum learning."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # The dataloader is re-created each epoch, so we can just read the epoch once.
        current_epoch = self._get_current_epoch()

        # Set random seed based on worker ID and current epoch
        # This ensures different shuffling each epoch while maintaining reproducibility
        try:
            base_seed = torch.initial_seed()
        except RuntimeError:
            base_seed = 42
        seed = (base_seed + current_epoch * num_workers + worker_id) % (2**32)
        random.seed(seed)
        np.random.seed(seed)

        while True:
            # WebDataset's with_epoch(nsamples) sets epoch length; it is not an
            # epoch-number notification API. Keep epoch handling in the shared
            # curriculum counter and pass splitters explicitly for DDP/workers.
            dataset = wds.WebDataset(
                self.shard_pattern,
                shardshuffle=(100 if self.shuffle else False),
                nodesplitter=wds.split_by_node if self.enable_node_worker_split else wds.single_node_only,
                workersplitter=wds.split_by_worker if self.enable_node_worker_split else _identity_splitter,
                seed=seed,
            )
            if self.shuffle:
                dataset = dataset.shuffle(self.shuffle_buffer_size)
            dataset = dataset.decode()

            # Process each game
            for sample in dataset:
                # Generate position pairs from the sample
                yield from self._generate_pairs(sample["positions.npy"], sample.get("game_id.txt", "unknown"))
            if not self.repeat:
                break

    def _generate_pairs(self, positions: NDArray[np.int32], game_id: str) -> Iterator[dict[str, Any]]:
        """Generate variable-distance position pairs from a game.

        Args:
            positions: Array of encoded positions [num_positions, 96]
            game_id: Game identifier

        Yields:
            Dictionary with target_tokens, source_tokens, distance, timestep, game_phase, and absolute_move
        """
        num_positions = len(positions)

        if num_positions < 2:
            return

        # Get current weights for curriculum learning
        current_weights = self._get_current_weights()

        # Generate pairs for each distance
        for i in range(num_positions):
            for dist_idx, distance in enumerate(self.distance_schedule.distances):
                if i + distance >= num_positions:
                    continue

                # Probabilistic sampling based on curriculum weights
                if random.random() >= current_weights[dist_idx]:
                    continue

                # For backward diffusion: target is earlier, source is later
                target_idx = i
                source_idx = i + distance

                target_tokens = positions[target_idx]
                source_tokens = positions[source_idx]

                # Apply augmentation
                if self.augment and random.random() < self.flip_prob:
                    target_tokens = self._flip_tokens(target_tokens)
                    source_tokens = self._flip_tokens(source_tokens)

                # Game phase: 0.0 (opening) to 1.0 (endgame)
                game_phase = i / max(1, num_positions - 1)

                result: dict[str, Any] = {
                    "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
                    "source_tokens": torch.tensor(source_tokens, dtype=torch.long),
                    "distance": torch.tensor(distance, dtype=torch.long),
                    "timestep": torch.tensor(distance, dtype=torch.long),  # timestep = distance
                    "game_phase": torch.tensor(game_phase, dtype=torch.float32),
                    "absolute_move": torch.tensor(i, dtype=torch.long),
                    "game_id": str(game_id),
                }
                yield result

    def _flip_tokens(self, tokens: NDArray[np.int32]) -> NDArray[np.int32]:
        """Apply horizontal flip to tokens."""
        return tokens[self._flip_indices]


def create_webdataset_dataloader(
    shard_pattern: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    distance_schedule: DistanceSchedule | None = None,
    shared_epoch: Any | None = None,
    shuffle_buffer_size: int = 1000,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    repeat: bool = False,
    drop_last: bool = True,
    **dataset_kwargs: Any,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    """Create DataLoader for WebDataset.

    Args:
        shard_pattern: Pattern for shard files
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle
        augment: Whether to augment
        distance_schedule: DistanceSchedule instance (required for ShogiWebDataset)
        shared_epoch: Multiprocessing-safe epoch counter
        shuffle_buffer_size: Size of the shuffle buffer for WebDataset
        pin_memory: Whether to use pinned memory
        persistent_workers: Whether to use persistent workers
        repeat: Whether to repeat the dataset indefinitely (useful for training)
        **dataset_kwargs: Additional arguments for ShogiWebDataset

    Returns:
        DataLoader instance
    """
    if distance_schedule is None:
        raise ValueError("distance_schedule is required for ShogiWebDataset")

    dataset = ShogiWebDataset(
        shard_pattern=shard_pattern,
        distance_schedule=distance_schedule,
        shuffle=shuffle,
        augment=augment,
        shared_epoch=shared_epoch,
        shuffle_buffer_size=shuffle_buffer_size,
        repeat=repeat,
        **dataset_kwargs,
    )

    # For IterableDataset, we don't use shuffle in DataLoader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load metadata for WebDataset.

    Args:
        metadata_path: Path to metadata.json

    Returns:
        Metadata dictionary
    """
    with Path(metadata_path).open() as f:
        return json.load(f)  # type: ignore[no-any-return]
