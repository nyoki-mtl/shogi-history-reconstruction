"""Build WebDataset from KIF files."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import webdataset as wds
from tqdm import tqdm

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.utils.kif import KIFParser


class WebDatasetBuilder:
    """Build WebDataset shards from KIF files."""

    def __init__(
        self,
        kif_directory: Path,
        output_directory: Path,
        shard_size: int = 10000,
        min_trajectory_length: int = 1,
        max_trajectory_length: int = 100,
    ):
        """Initialize WebDataset builder.

        Args:
            kif_directory: Directory containing KIF files
            output_directory: Directory to save WebDataset shards
            shard_size: Number of games per shard
            min_trajectory_length: Minimum number of moves in a game
            max_trajectory_length: Maximum number of moves to use from a game
        """
        self.kif_directory = Path(kif_directory)
        self.output_directory = Path(output_directory)
        self.shard_size = shard_size
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.parser: KIFParser = KIFParser()
        self.tokenizer = ShogiTokenizer(flip_to_black=False)

    def build_shards(self) -> dict[str, Any]:
        """Build WebDataset shards from KIF files.

        Returns:
            Metadata about the created shards
        """
        kif_files = sorted(self.kif_directory.glob("*.kif"))
        print(f"Found {len(kif_files)} KIF files")

        shard_idx = 0
        games_in_shard = 0
        total_games = 0
        total_pairs = 0
        shard_pattern = str(self.output_directory / "shard_%05d.tar").replace("\\", "/")
        if len(shard_pattern) >= 3 and shard_pattern[1:3] == ":/":
            shard_pattern = f"file:{shard_pattern}"
        current_sink = None
        shard_metadata = []

        for _file_idx, kif_file in enumerate(tqdm(kif_files, desc="Processing KIF files")):
            # Parse KIF file
            game_data = self.parser.parse_kif_file(kif_file)
            if not game_data or len(game_data["moves"]) < self.min_trajectory_length:
                continue

            # Create new shard if needed
            if games_in_shard == 0:
                if current_sink:
                    current_sink.close()
                current_sink = wds.TarWriter(shard_pattern % shard_idx)
                shard_metadata.append(
                    {
                        "shard_idx": shard_idx,
                        "filename": f"shard_{shard_idx:05d}.tar",
                        "games": 0,
                        "pairs": 0,
                    }
                )

            # Process game into position pairs
            pairs_data = self._process_game(game_data)

            sample = {
                "__key__": f"{shard_idx:05d}_{games_in_shard:05d}",
                "game_id.txt": kif_file.stem.encode(),
                "positions.npy": pairs_data["positions"],
                "win.json": json.dumps(pairs_data["win"]).encode(),
                "endgame.txt": pairs_data["endgame"].encode(),
                "num_moves.json": json.dumps(pairs_data["num_moves"]).encode(),
            }
            if current_sink is not None:
                current_sink.write(sample)

            games_in_shard += 1
            total_games += 1
            pairs_count = len(pairs_data["positions"]) - 1
            total_pairs += pairs_count
            games_val = shard_metadata[-1].get("games", 0)
            pairs_val = shard_metadata[-1].get("pairs", 0)
            shard_metadata[-1]["games"] = (games_val if isinstance(games_val, int) else 0) + 1
            shard_metadata[-1]["pairs"] = (pairs_val if isinstance(pairs_val, int) else 0) + pairs_count

            # Check if shard is full
            if games_in_shard >= self.shard_size:
                shard_idx += 1
                games_in_shard = 0

        # Close last shard
        if current_sink:
            current_sink.close()

        # Write metadata
        metadata = {
            "total_shards": len(shard_metadata),
            "total_games": total_games,
            "total_pairs": total_pairs,
            "shard_size": self.shard_size,
            "min_trajectory_length": self.min_trajectory_length,
            "max_trajectory_length": self.max_trajectory_length,
            "shards": shard_metadata,
        }

        metadata_path = self.output_directory / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nCreated {len(shard_metadata)} shards")
        print(f"Total games: {total_games}")
        print(f"Total position pairs: {total_pairs}")
        print(f"Metadata saved to: {metadata_path}")

        return metadata

    def _process_game(self, game_data: dict[str, Any]) -> dict[str, Any]:
        """Process a single game into position pairs.

        Args:
            game_data: Parsed game data
        Returns:
            Processed game data with encoded positions
        """
        board = self.parser.board
        board.reset()
        positions = [self.tokenizer.encode(board)]

        for i, move in enumerate(game_data["moves"]):
            if i >= self.max_trajectory_length or move == 0:
                break
            board.push_move(move)
            positions.append(self.tokenizer.encode(board))

        return {
            "positions": np.array(positions, dtype=np.int32),
            "win": game_data["win"],
            "endgame": game_data["endgame"],
            "num_moves": len(positions) - 1,
        }


def create_webdataset_from_kif(
    kif_directory: str | Path,
    output_directory: str | Path,
    shard_size: int = 10000,
    min_moves: int = 1,
    max_moves: int = 100,
) -> None:
    """Convenience function to create WebDataset from KIF files.

    Args:
        kif_directory: Directory containing KIF files
        output_directory: Directory to save WebDataset shards
        shard_size: Number of games per shard
        min_moves: Minimum number of moves per game
        max_moves: Maximum number of moves per game
    """
    builder = WebDatasetBuilder(
        kif_directory=Path(kif_directory),
        output_directory=Path(output_directory),
        shard_size=shard_size,
        min_trajectory_length=min_moves,
        max_trajectory_length=max_moves,
    )
    builder.build_shards()
