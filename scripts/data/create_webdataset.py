#!/usr/bin/env python3
"""Create WebDataset from KIF files."""

import argparse
from pathlib import Path

from posdiffusion.data.webdataset_builder import create_webdataset_from_kif


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Create WebDataset from KIF files")
    parser.add_argument(
        "--kif-dir",
        type=Path,
        default=Path("data/kif"),
        help="Directory containing KIF files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/shards"),
        help="Output directory for WebDataset shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of games per shard",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=1,
        help="Minimum number of moves per game",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=100,
        help="Maximum number of moves per game",
    )
    args = parser.parse_args()

    print(f"Creating WebDataset from {args.kif_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Shard size: {args.shard_size}")
    print(f"Min moves: {args.min_moves}")
    print(f"Max moves: {args.max_moves}")

    create_webdataset_from_kif(
        kif_directory=args.kif_dir,
        output_directory=args.output_dir,
        shard_size=args.shard_size,
        min_moves=args.min_moves,
        max_moves=args.max_moves,
    )


if __name__ == "__main__":
    main()
