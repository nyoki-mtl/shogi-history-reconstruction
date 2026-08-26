"""Test variable distance data loader functionality."""

import io
import tarfile

import numpy as np
import webdataset as wds

from posdiffusion.data.webdataset_loader import ShogiWebDataset
from posdiffusion.diffusion.schedules import DistanceSchedule


def _npy_payload(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _add_tar_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _write_webdataset_shard(path, *, num_games: int = 4, num_positions: int = 3) -> None:
    with tarfile.open(path, "w") as tar:
        for game_idx in range(num_games):
            key = f"{game_idx:06d}"
            positions = np.full((num_positions, 96), fill_value=game_idx, dtype=np.int32)
            positions[:, 95] = np.arange(num_positions, dtype=np.int32) % 2
            _add_tar_member(tar, f"{key}.positions.npy", _npy_payload(positions))
            _add_tar_member(tar, f"{key}.game_id.txt", f"game-{game_idx}".encode())


class TestVariableDistanceLoader:
    """Test variable distance functionality in ShogiWebDataset."""

    def test_distance_config_initialization(self):
        """Test that distance config is properly initialized."""
        from posdiffusion.diffusion.schedules import DistanceSchedule

        distance_schedule = DistanceSchedule(
            num_steps=10,
            distances=[1, 3, 5],
            weights=[0.5, 0.3, 0.2],
            curriculum_schedule={
                0: [0.8, 0.15, 0.05],
                3: [0.4, 0.4, 0.2],
            },
        )

        dataset = ShogiWebDataset(
            shard_pattern="dummy_pattern",
            distance_schedule=distance_schedule,
        )

        assert dataset.distance_schedule.distances == [1, 3, 5]
        assert dataset.distance_schedule.weights == [0.5, 0.3, 0.2]
        assert 0 in dataset.distance_schedule.curriculum_schedule
        assert 3 in dataset.distance_schedule.curriculum_schedule

    def test_curriculum_weight_selection(self):
        """Test curriculum learning weight selection based on epoch."""
        from posdiffusion.diffusion.schedules import DistanceSchedule

        distance_schedule = DistanceSchedule(
            num_steps=10,
            distances=[1, 2, 5],
            weights=[0.33, 0.33, 0.34],
            curriculum_schedule={
                0: [0.7, 0.2, 0.1],
                2: [0.4, 0.4, 0.2],
                5: [0.33, 0.33, 0.34],
            },
        )

        # The dataset uses shared_epoch to track current epoch; pass a simple holder
        class EpochHolder:
            def __init__(self, value: int):
                self.value = value

        epoch = EpochHolder(0)
        dataset = ShogiWebDataset(
            shard_pattern="dummy_pattern",
            distance_schedule=distance_schedule,
            shared_epoch=epoch,
        )

        # Epoch 0: should use first schedule
        weights = dataset._get_current_weights()
        assert weights == [0.7, 0.2, 0.1]

        # Epoch 1: should still use epoch 0 schedule
        epoch.value = 1
        weights = dataset._get_current_weights()
        assert weights == [0.7, 0.2, 0.1]

        # Epoch 2: should use epoch 2 schedule
        epoch.value = 2
        weights = dataset._get_current_weights()
        assert weights == [0.4, 0.4, 0.2]

        # Epoch 6: should use epoch 5 schedule
        epoch.value = 6
        weights = dataset._get_current_weights()
        assert weights == [0.33, 0.33, 0.34]

    def test_generate_pairs_variable_distance(self):
        """Test generation of variable distance pairs."""
        distance_schedule = DistanceSchedule(
            num_steps=5,
            distances=[1, 2, 3],
            weights=[1.0, 1.0, 1.0],  # Always generate for testing
            curriculum_schedule={},
        )

        dataset = ShogiWebDataset(
            shard_pattern="dummy_pattern",
            distance_schedule=distance_schedule,
            augment=False,  # Disable augmentation for predictable testing
        )

        # Create dummy game positions
        num_positions = 10
        positions = np.random.randint(0, 40, size=(num_positions, 96), dtype=np.int32)

        # Collect all generated pairs
        pairs = list(dataset._generate_pairs(positions, "test_game"))

        # Check that pairs were generated for different distances
        distances = [p["distance"].item() for p in pairs]
        assert 1 in distances
        assert 2 in distances
        assert 3 in distances

        # Check pair properties
        for pair in pairs:
            assert "target_tokens" in pair
            assert "source_tokens" in pair
            assert "distance" in pair
            assert "timestep" in pair
            assert "game_phase" in pair
            assert "absolute_move" in pair
            assert "game_id" in pair

            # Verify timestep equals distance
            assert pair["timestep"].item() == pair["distance"].item()

            # Verify game phase is in [0, 1]
            assert 0.0 <= pair["game_phase"].item() <= 1.0

            # Verify absolute move is valid
            assert 0 <= pair["absolute_move"].item() < num_positions

    def test_iter_reads_all_games_when_epoch_changes(self, tmp_path, monkeypatch):
        """Changing the curriculum epoch must not truncate WebDataset samples."""

        def fail_with_epoch(*args, **kwargs):
            raise AssertionError("with_epoch must not be used for epoch tracking")

        monkeypatch.setattr(wds.DataPipeline, "with_epoch", fail_with_epoch)

        class EpochHolder:
            def __init__(self, value: int):
                self.value = value

        shard_path = tmp_path / "sample.tar"
        _write_webdataset_shard(shard_path, num_games=4, num_positions=3)
        distance_schedule = DistanceSchedule(
            num_steps=1,
            distances=[1],
            weights=[1.0],
            curriculum_schedule={0: [1.0], 3: [1.0]},
        )
        epoch = EpochHolder(3)
        dataset = ShogiWebDataset(
            shard_pattern=str(shard_path),
            distance_schedule=distance_schedule,
            shuffle=False,
            augment=False,
            shared_epoch=epoch,
            enable_node_worker_split=False,
        )

        pairs = list(dataset)
        assert len(pairs) == 8
        assert {pair["game_id"] for pair in pairs} == {"game-0", "game-1", "game-2", "game-3"}

        epoch.value = 9
        pairs_after_epoch_change = list(dataset)
        assert len(pairs_after_epoch_change) == 8
        assert {pair["game_id"] for pair in pairs_after_epoch_change} == {"game-0", "game-1", "game-2", "game-3"}
