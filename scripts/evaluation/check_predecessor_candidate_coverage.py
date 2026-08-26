#!/usr/bin/env python3
"""Audit legal predecessor candidate coverage on WebDataset shards."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import webdataset as wds
from omegaconf import OmegaConf

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.reverse.candidates import CandidateBackend, generate_predecessor_candidates


@dataclass
class CoverageRow:
    game_id: str
    move_index: int
    move_bin: str
    candidate_count: int
    true_predecessor_covered: bool
    drop_candidate_count: int
    capture_candidate_count: int
    promotion_candidate_count: int
    nonpromotion_move_candidate_count: int
    successor_side_to_move_in_check: bool
    true_rank_rule_order: int | None
    true_kind: str | None
    true_is_drop: bool | None
    true_is_capture: bool | None
    true_is_promotion: bool | None


@dataclass
class GroupSummaryRow:
    group: str
    value: str
    num_positions: int
    true_predecessor_coverage: float
    candidate_count_mean: float
    candidate_count_median: float
    candidate_count_p90: float
    candidate_count_p99: float
    drop_candidate_mean: float
    capture_candidate_mean: float
    promotion_candidate_mean: float
    nonpromotion_move_candidate_mean: float


def load_shard_pattern(config_path: str | Path, split: str) -> str:
    cfg = OmegaConf.load(config_path)
    pattern = OmegaConf.select(cfg, f"data.{split}_shard_pattern") or OmegaConf.select(cfg, f"{split}_shard_pattern")
    if pattern is None:
        raise ValueError(f"Config {config_path} must contain data.{split}_shard_pattern")
    return str(pattern)


def evaluate_game(
    *,
    tokenizer: ShogiTokenizer,
    game_id: str,
    positions: np.ndarray,
    max_positions: int,
    candidate_backend: CandidateBackend,
    move_bin_size: int,
) -> list[CoverageRow]:
    rows: list[CoverageRow] = []
    end = len(positions)
    if max_positions > 0:
        end = min(end, max_positions + 1)

    for move_idx in range(1, end):
        predecessor = tuple(int(token) for token in positions[move_idx - 1])
        successor = positions[move_idx]
        candidates = generate_predecessor_candidates(successor, tokenizer=tokenizer, backend=candidate_backend)
        candidate_tokens = [candidate.tokens for candidate in candidates]
        true_index = candidate_tokens.index(predecessor) if predecessor in set(candidate_tokens) else None
        true_candidate = candidates[true_index] if true_index is not None else None
        successor_board = tokenizer.decode([int(token) for token in successor])
        if move_bin_size > 0:
            bin_start = ((move_idx - 1) // move_bin_size) * move_bin_size + 1
            bin_end = bin_start + move_bin_size - 1
            move_bin = f"{bin_start:03d}-{bin_end:03d}"
        else:
            move_bin = f"{move_idx:03d}"
        rows.append(
            CoverageRow(
                game_id=game_id,
                move_index=move_idx,
                move_bin=move_bin,
                candidate_count=len(candidates),
                true_predecessor_covered=true_index is not None,
                drop_candidate_count=sum(1 for candidate in candidates if candidate.kind == "drop"),
                capture_candidate_count=sum(1 for candidate in candidates if candidate.captured_piece is not None),
                promotion_candidate_count=sum(1 for candidate in candidates if candidate.promoted),
                nonpromotion_move_candidate_count=sum(
                    1 for candidate in candidates if candidate.kind == "move" and not candidate.promoted
                ),
                successor_side_to_move_in_check=successor_board.is_in_check(),
                true_rank_rule_order=true_index + 1 if true_index is not None else None,
                true_kind=true_candidate.kind if true_candidate is not None else None,
                true_is_drop=true_candidate.kind == "drop" if true_candidate is not None else None,
                true_is_capture=true_candidate.captured_piece is not None if true_candidate is not None else None,
                true_is_promotion=true_candidate.promoted if true_candidate is not None else None,
            )
        )
    return rows


def summarize_group(group: str, value: str, rows: list[CoverageRow]) -> GroupSummaryRow:
    counts = np.asarray([row.candidate_count for row in rows], dtype=np.int64)
    covered = np.asarray([row.true_predecessor_covered for row in rows], dtype=np.bool_)
    return GroupSummaryRow(
        group=group,
        value=value,
        num_positions=len(rows),
        true_predecessor_coverage=float(covered.mean()) if len(rows) else 0.0,
        candidate_count_mean=float(counts.mean()) if len(rows) else 0.0,
        candidate_count_median=float(np.median(counts)) if len(rows) else 0.0,
        candidate_count_p90=float(np.quantile(counts, 0.9)) if len(rows) else 0.0,
        candidate_count_p99=float(np.quantile(counts, 0.99)) if len(rows) else 0.0,
        drop_candidate_mean=float(np.mean([row.drop_candidate_count for row in rows])) if rows else 0.0,
        capture_candidate_mean=float(np.mean([row.capture_candidate_count for row in rows])) if rows else 0.0,
        promotion_candidate_mean=float(np.mean([row.promotion_candidate_count for row in rows])) if rows else 0.0,
        nonpromotion_move_candidate_mean=(
            float(np.mean([row.nonpromotion_move_candidate_count for row in rows])) if rows else 0.0
        ),
    )


def group_rows(rows: list[CoverageRow]) -> list[GroupSummaryRow]:
    grouped: list[GroupSummaryRow] = []
    group_specs = {
        "move_bin": lambda row: row.move_bin,
        "true_kind": lambda row: row.true_kind or "uncovered",
        "true_capture": lambda row: "capture" if row.true_is_capture else "non_capture",
        "true_promotion": lambda row: "promotion" if row.true_is_promotion else "non_promotion",
        "successor_check": lambda row: "check" if row.successor_side_to_move_in_check else "no_check",
    }
    for group, key_fn in group_specs.items():
        buckets: dict[str, list[CoverageRow]] = {}
        for row in rows:
            buckets.setdefault(key_fn(row), []).append(row)
        for value in sorted(buckets):
            grouped.append(summarize_group(group, value, buckets[value]))
    return grouped


def aggregate(rows: list[CoverageRow]) -> dict[str, Any]:
    if not rows:
        return {"num_positions": 0}

    counts = np.asarray([row.candidate_count for row in rows], dtype=np.int64)
    covered = np.asarray([row.true_predecessor_covered for row in rows], dtype=np.bool_)
    summary: dict[str, Any] = {
        "num_positions": len(rows),
        "true_predecessor_coverage": float(covered.mean()),
        "candidate_count_mean": float(counts.mean()),
        "candidate_count_median": float(np.median(counts)),
        "candidate_count_p90": float(np.quantile(counts, 0.9)),
        "candidate_count_p99": float(np.quantile(counts, 0.99)),
        "candidate_count_min": int(counts.min()),
        "candidate_count_max": int(counts.max()),
        "zero_candidate_count": int((counts == 0).sum()),
        "drop_candidate_mean": float(np.mean([row.drop_candidate_count for row in rows])),
        "capture_candidate_mean": float(np.mean([row.capture_candidate_count for row in rows])),
        "promotion_candidate_mean": float(np.mean([row.promotion_candidate_count for row in rows])),
        "nonpromotion_move_candidate_mean": float(np.mean([row.nonpromotion_move_candidate_count for row in rows])),
        "successor_check_rate": float(np.mean([row.successor_side_to_move_in_check for row in rows])),
        "true_drop_rate_covered": float(np.mean([row.true_is_drop for row in rows if row.true_is_drop is not None])),
        "true_capture_rate_covered": float(
            np.mean([row.true_is_capture for row in rows if row.true_is_capture is not None])
        ),
        "true_promotion_rate_covered": float(
            np.mean([row.true_is_promotion for row in rows if row.true_is_promotion is not None])
        ),
    }
    uncovered = [row for row in rows if not row.true_predecessor_covered]
    summary["uncovered_count"] = len(uncovered)
    summary["uncovered_examples"] = [
        {"game_id": row.game_id, "move_index": row.move_index, "candidate_count": row.candidate_count}
        for row in uncovered[:20]
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit legal predecessor candidate coverage")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--skip_games", "--game_start", dest="skip_games", type=int, default=0)
    parser.add_argument("--max_games", type=int, default=10, help="0 means no limit")
    parser.add_argument("--game_limit", dest="max_games", type=int)
    parser.add_argument("--max_positions_per_game", type=int, default=20, help="0 means all")
    parser.add_argument("--csv_limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--output_dir", type=str, default="work_dir/evaluation/predecessor_candidate_coverage")
    parser.add_argument("--candidate_backend", choices=["rsshogi", "yaneuraou", "yaneuraou_hybrid"], default="rsshogi")
    parser.add_argument("--move_bin_size", type=int, default=10, help="0 means exact move index")
    parser.add_argument("--progress_every_games", type=int, default=0, help="0 disables progress logging")
    args = parser.parse_args()
    if args.skip_games < 0:
        raise ValueError("--skip_games must be non-negative")
    if args.move_bin_size < 0:
        raise ValueError("--move_bin_size must be non-negative")

    tokenizer = ShogiTokenizer(flip_to_black=False)
    shard_pattern = load_shard_pattern(args.config, args.split)
    dataset = wds.WebDataset(shard_pattern, shardshuffle=False).decode()

    rows: list[CoverageRow] = []
    game_count = 0
    started = time.time()
    for sample_idx, sample in enumerate(dataset):
        if sample_idx < args.skip_games:
            continue
        game_id = sample.get("game_id.txt", sample.get("__key__", "unknown"))
        rows.extend(
            evaluate_game(
                tokenizer=tokenizer,
                game_id=str(game_id),
                positions=sample["positions.npy"],
                max_positions=args.max_positions_per_game,
                candidate_backend=args.candidate_backend,
                move_bin_size=args.move_bin_size,
            )
        )
        game_count += 1
        if args.progress_every_games > 0 and game_count % args.progress_every_games == 0:
            elapsed = time.time() - started
            print(
                "Progress: "
                f"games={game_count} rows={len(rows)} elapsed={elapsed:.1f}s "
                f"sec_per_game={elapsed / game_count:.2f}",
                flush=True,
            )
        if args.max_games > 0 and game_count >= args.max_games:
            break

    summary = aggregate(rows)
    summary.update(
        {
            "split": args.split,
            "shard_pattern": shard_pattern,
            "num_games": game_count,
            "skip_games": args.skip_games,
            "max_games": args.max_games,
            "max_positions_per_game": args.max_positions_per_game,
            "candidate_backend": args.candidate_backend,
            "move_bin_size": args.move_bin_size,
            "elapsed_seconds": time.time() - started,
            "seconds_per_game": (time.time() - started) / game_count if game_count else None,
            "seconds_per_position": (time.time() - started) / len(rows) if rows else None,
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"predecessor_candidate_coverage_{args.split}_{timestamp}.csv"
    group_csv_path = out_dir / f"predecessor_candidate_coverage_groups_{args.split}_{timestamp}.csv"
    json_path = out_dir / f"predecessor_candidate_coverage_{args.split}_summary_{timestamp}.json"

    limit = args.csv_limit if args.csv_limit > 0 else None
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CoverageRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for idx, row in enumerate(rows):
            if limit is not None and idx >= limit:
                break
            writer.writerow(asdict(row))

    grouped = group_rows(rows)
    with group_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(GroupSummaryRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in grouped:
            writer.writerow(asdict(row))

    summary["group_summaries"] = [asdict(row) for row in grouped]
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed games: {game_count}")
    print(f"Positions evaluated: {len(rows)}")
    print(f"Summary saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Group CSV saved to: {group_csv_path}")


if __name__ == "__main__":
    main()
