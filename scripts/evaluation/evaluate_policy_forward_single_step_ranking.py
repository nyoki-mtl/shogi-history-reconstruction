#!/usr/bin/env python3
"""Evaluate single-step predecessor ranking under a forward policy model.

For each sampled successor position, this script enumerates legal predecessor
candidates and ranks them by the normalized AlphaZero policy probability of the
forward move from the candidate predecessor into the successor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import webdataset as wds
from omegaconf import OmegaConf
from rsshogi.core import Move
from rsshogi.policy import move_label

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.evaluation.policy.alphazero_onnx import AlphaZeroONNX
from posdiffusion.reverse.candidates import CandidateBackend, PredecessorCandidate, generate_predecessor_candidates
from posdiffusion.reverse.scoring import true_candidate_index


@dataclass(frozen=True)
class MoveBin:
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class SampledPosition:
    bin_label: str
    game_id: str
    move_index: int
    predecessor_tokens: tuple[int, ...]
    successor_tokens: tuple[int, ...]


@dataclass
class RankingRow:
    bin_label: str
    game_id: str
    move_index: int
    candidate_count: int
    true_predecessor_covered: bool
    true_rank: int | None
    top1_exact: bool
    top3_exact: bool
    top5_exact: bool
    selected_move_usi: str | None
    true_move_usi: str | None
    true_policy_prob: float | None
    best_policy_prob: float | None
    true_log_score: float | None
    best_log_score: float | None
    evaluation_seconds: float


def parse_bins(raw: str) -> list[MoveBin]:
    bins: list[MoveBin] = []
    seen: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.replace("--", "-").split("-", maxsplit=1)
            start = int(start_raw)
            end = int(end_raw)
        else:
            start = int(item)
            end = start
        if start <= 0 or end < start:
            raise ValueError(f"Invalid move bin: {item}")
        for move_index in range(start, end + 1):
            if move_index in seen:
                raise ValueError(f"Overlapping move bins include move {move_index}")
            seen.add(move_index)
        label = f"{start}--{end}" if start != end else str(start)
        bins.append(MoveBin(label=label, start=start, end=end))
    if not bins:
        raise ValueError("At least one move bin is required")
    return bins


def load_shard_pattern(config_path: str | Path, split: str) -> str:
    cfg = OmegaConf.load(config_path)
    pattern = OmegaConf.select(cfg, f"data.{split}_shard_pattern") or OmegaConf.select(cfg, f"{split}_shard_pattern")
    if pattern is None:
        raise ValueError(f"Config {config_path} must contain data.{split}_shard_pattern")
    return str(pattern)


def build_move_to_bin(bins: list[MoveBin]) -> dict[int, str]:
    return {move_index: move_bin.label for move_bin in bins for move_index in range(move_bin.start, move_bin.end + 1)}


def reservoir_add(
    *,
    reservoirs: dict[str, list[SampledPosition]],
    seen_counts: dict[str, int],
    item: SampledPosition,
    samples_per_bin: int,
    rng: random.Random,
) -> None:
    seen_counts[item.bin_label] += 1
    reservoir = reservoirs[item.bin_label]
    if len(reservoir) < samples_per_bin:
        reservoir.append(item)
        return
    replacement_index = rng.randrange(seen_counts[item.bin_label])
    if replacement_index < samples_per_bin:
        reservoir[replacement_index] = item


def sample_positions(
    *,
    shard_pattern: str,
    bins: list[MoveBin],
    samples_per_bin: int,
    seed: int,
    max_games: int,
) -> tuple[list[SampledPosition], dict[str, int], int]:
    rng = random.Random(seed)
    move_to_bin = build_move_to_bin(bins)
    target_moves = sorted(move_to_bin)
    reservoirs = {move_bin.label: [] for move_bin in bins}
    seen_counts = {move_bin.label: 0 for move_bin in bins}
    dataset = wds.WebDataset(shard_pattern, shardshuffle=False).decode()

    games_seen = 0
    for sample in dataset:
        game_id = str(sample.get("game_id.txt", sample.get("__key__", "unknown")))
        positions = sample["positions.npy"]
        for move_index in target_moves:
            if move_index >= len(positions):
                continue
            bin_label = move_to_bin[move_index]
            item = SampledPosition(
                bin_label=bin_label,
                game_id=game_id,
                move_index=move_index,
                predecessor_tokens=tuple(int(token) for token in positions[move_index - 1]),
                successor_tokens=tuple(int(token) for token in positions[move_index]),
            )
            reservoir_add(
                reservoirs=reservoirs,
                seen_counts=seen_counts,
                item=item,
                samples_per_bin=samples_per_bin,
                rng=rng,
            )
        games_seen += 1
        if max_games > 0 and games_seen >= max_games:
            break

    selected: list[SampledPosition] = []
    for move_bin in bins:
        selected.extend(reservoirs[move_bin.label])
    return selected, seen_counts, games_seen


def policy_probabilities(
    *,
    alphazero: AlphaZeroONNX,
    tokenizer: ShogiTokenizer,
    candidates: list[PredecessorCandidate],
    batch_size: int,
) -> list[float]:
    if not candidates:
        return []
    predecessor_boards = [tokenizer.decode(list(candidate.tokens)) for candidate in candidates]
    probabilities: list[float] = []
    for offset in range(0, len(predecessor_boards), batch_size):
        board_batch = predecessor_boards[offset : offset + batch_size]
        candidate_batch = candidates[offset : offset + batch_size]
        policy_probs_batch, _ = alphazero.get_policy_value_batch(board_batch)
        for candidate, predecessor_board, policy_probs in zip(
            candidate_batch, board_batch, policy_probs_batch, strict=True
        ):
            legal_total = 0.0
            for legal_move in predecessor_board.legal_moves():
                legal_total += float(policy_probs[move_label(legal_move, predecessor_board.turn)])
            move = Move.from_usi(candidate.move_usi)
            probability = float(policy_probs[move_label(move, predecessor_board.turn)])
            if legal_total > 0.0:
                probability /= legal_total
            probabilities.append(probability)
    return probabilities


def evaluate_position(
    *,
    item: SampledPosition,
    tokenizer: ShogiTokenizer,
    alphazero: AlphaZeroONNX,
    candidate_backend: CandidateBackend,
    policy_batch_size: int,
) -> RankingRow:
    started = time.time()
    candidates = generate_predecessor_candidates(
        np.asarray(item.successor_tokens, dtype=np.int64),
        tokenizer=tokenizer,
        backend=candidate_backend,
    )
    candidate_count = len(candidates)
    true_idx = true_candidate_index(candidates, item.predecessor_tokens)
    covered = true_idx is not None

    if candidate_count == 0:
        return RankingRow(
            bin_label=item.bin_label,
            game_id=item.game_id,
            move_index=item.move_index,
            candidate_count=0,
            true_predecessor_covered=False,
            true_rank=None,
            top1_exact=False,
            top3_exact=False,
            top5_exact=False,
            selected_move_usi=None,
            true_move_usi=None,
            true_policy_prob=None,
            best_policy_prob=None,
            true_log_score=None,
            best_log_score=None,
            evaluation_seconds=time.time() - started,
        )

    probabilities = policy_probabilities(
        alphazero=alphazero,
        tokenizer=tokenizer,
        candidates=candidates,
        batch_size=policy_batch_size,
    )
    scores = [math.log(max(probability, 1.0e-45)) - index * 1.0e-9 for index, probability in enumerate(probabilities)]
    best_idx = max(range(candidate_count), key=lambda index: scores[index])

    true_rank: int | None = None
    top1_exact = False
    top3_exact = False
    top5_exact = False
    true_policy_prob: float | None = None
    true_log_score: float | None = None
    true_move_usi: str | None = None
    if true_idx is not None:
        true_score = scores[true_idx]
        true_rank = 1 + sum(score > true_score for score in scores)
        top1_exact = best_idx == true_idx
        top3_exact = true_rank <= 3
        top5_exact = true_rank <= 5
        true_policy_prob = probabilities[true_idx]
        true_log_score = true_score
        true_move_usi = candidates[true_idx].move_usi

    return RankingRow(
        bin_label=item.bin_label,
        game_id=item.game_id,
        move_index=item.move_index,
        candidate_count=candidate_count,
        true_predecessor_covered=covered,
        true_rank=true_rank,
        top1_exact=top1_exact,
        top3_exact=top3_exact,
        top5_exact=top5_exact,
        selected_move_usi=candidates[best_idx].move_usi,
        true_move_usi=true_move_usi,
        true_policy_prob=true_policy_prob,
        best_policy_prob=probabilities[best_idx],
        true_log_score=true_log_score,
        best_log_score=scores[best_idx],
        evaluation_seconds=time.time() - started,
    )


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize_rows(rows: list[RankingRow], seen_count: int | None = None) -> dict[str, Any]:
    covered = [row for row in rows if row.true_predecessor_covered]
    candidate_counts = [float(row.candidate_count) for row in rows]
    ranks = [float(row.true_rank) for row in covered if row.true_rank is not None]
    true_probs = [row.true_policy_prob for row in covered if row.true_policy_prob is not None]
    best_probs = [row.best_policy_prob for row in rows if row.best_policy_prob is not None]
    eval_seconds = [row.evaluation_seconds for row in rows]
    random_terms = [1.0 / row.candidate_count for row in covered if row.candidate_count > 0]
    return {
        "num_positions": len(rows),
        "seen_positions": seen_count,
        "true_predecessor_coverage": len(covered) / len(rows) if rows else 0.0,
        "top1_exact": sum(row.top1_exact for row in rows) / len(rows) if rows else 0.0,
        "top3_exact": sum(row.top3_exact for row in rows) / len(rows) if rows else 0.0,
        "top5_exact": sum(row.top5_exact for row in rows) / len(rows) if rows else 0.0,
        "covered_top1_exact": sum(row.top1_exact for row in covered) / len(covered) if covered else 0.0,
        "covered_top3_exact": sum(row.top3_exact for row in covered) / len(covered) if covered else 0.0,
        "covered_top5_exact": sum(row.top5_exact for row in covered) / len(covered) if covered else 0.0,
        "random_legal_top1_expected": float(np.mean(random_terms)) if random_terms else 0.0,
        "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "candidate_count_median": quantile(candidate_counts, 0.5),
        "candidate_count_p90": quantile(candidate_counts, 0.9),
        "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
        "true_rank_median": quantile(ranks, 0.5),
        "true_rank_p90": quantile(ranks, 0.9),
        "true_policy_prob_mean": float(np.mean(true_probs)) if true_probs else None,
        "true_policy_prob_median": quantile(true_probs, 0.5),
        "best_policy_prob_mean": float(np.mean(best_probs)) if best_probs else None,
        "best_policy_prob_median": quantile(best_probs, 0.5),
        "evaluation_seconds_mean": float(np.mean(eval_seconds)) if eval_seconds else 0.0,
        "evaluation_seconds_total": float(sum(eval_seconds)),
    }


def aggregate(rows: list[RankingRow], bins: list[MoveBin], seen_counts: dict[str, int]) -> dict[str, Any]:
    by_bin: dict[str, Any] = {}
    for move_bin in bins:
        selected = [row for row in rows if row.bin_label == move_bin.label]
        by_bin[move_bin.label] = summarize_rows(selected, seen_counts.get(move_bin.label))
    return {
        "overall": summarize_rows(rows, sum(seen_counts.values())),
        "by_bin": by_bin,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "| Move bin | N | Coverage | Top-1 | Top-3 | Top-5 | Rand. | Cand. mean | True prob mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, block in summary["by_bin"].items():
        true_prob = block["true_policy_prob_mean"]
        true_prob_text = f"{true_prob:.3f}" if true_prob is not None else "NA"
        lines.append(
            "| "
            f"{label} | {block['num_positions']} | {block['true_predecessor_coverage']:.3f} | "
            f"{block['top1_exact']:.3f} | {block['top3_exact']:.3f} | {block['top5_exact']:.3f} | "
            f"{block['random_legal_top1_expected']:.3f} | {block['candidate_count_mean']:.1f} | "
            f"{true_prob_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_policy_device() -> str:
    return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forward-policy single-step predecessor ranking")
    parser.add_argument("--alphazero_onnx", type=str, default="data/models/dl_suisho/model.onnx")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--bins", type=str, default="1-10,51-60,71-80,91-100")
    parser.add_argument("--samples_per_bin", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_games", type=int, default=0, help="0 means scan the full split")
    parser.add_argument("--policy_device", choices=["cpu", "cuda"], default=default_policy_device())
    parser.add_argument("--policy_intra_op_threads", type=int, default=4)
    parser.add_argument("--policy_batch_size", type=int, default=256)
    parser.add_argument(
        "--candidate_backend",
        choices=["rsshogi", "yaneuraou", "yaneuraou_hybrid"],
        default="yaneuraou",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="work_dir/evaluation/policy_forward_single_step_ranking",
    )
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    if args.samples_per_bin <= 0:
        raise ValueError("--samples_per_bin must be positive")
    if args.policy_batch_size <= 0:
        raise ValueError("--policy_batch_size must be positive")
    if args.policy_intra_op_threads <= 0:
        raise ValueError("--policy_intra_op_threads must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress_every must be positive")

    bins = parse_bins(args.bins)
    shard_pattern = load_shard_pattern(args.config, args.split)
    started = time.time()
    selected, seen_counts, games_seen = sample_positions(
        shard_pattern=shard_pattern,
        bins=bins,
        samples_per_bin=args.samples_per_bin,
        seed=args.seed,
        max_games=args.max_games,
    )
    print(
        f"Selected {len(selected)} positions from {games_seen} games; "
        f"seen per bin: {json.dumps(seen_counts, sort_keys=True)}",
        flush=True,
    )

    tokenizer = ShogiTokenizer(flip_to_black=False)
    alphazero = AlphaZeroONNX(
        args.alphazero_onnx,
        device=args.policy_device,
        intra_op_num_threads=args.policy_intra_op_threads,
    )

    rows: list[RankingRow] = []
    for index, item in enumerate(selected, start=1):
        rows.append(
            evaluate_position(
                item=item,
                tokenizer=tokenizer,
                alphazero=alphazero,
                candidate_backend=args.candidate_backend,
                policy_batch_size=args.policy_batch_size,
            )
        )
        if index % args.progress_every == 0 or index == len(selected):
            elapsed = time.time() - started
            print(
                f"Progress: positions={index}/{len(selected)} elapsed={elapsed:.1f}s "
                f"sec_per_position={elapsed / index:.2f}",
                flush=True,
            )

    summary = aggregate(rows, bins, seen_counts)
    summary.update(
        {
            "script": str(Path(__file__)),
            "command": " ".join(sys.argv),
            "split": args.split,
            "shard_pattern": shard_pattern,
            "alphazero_onnx": args.alphazero_onnx,
            "onnx_available_providers": ort.get_available_providers(),
            "policy_device_requested": args.policy_device,
            "policy_intra_op_threads": args.policy_intra_op_threads,
            "policy_batch_size": args.policy_batch_size,
            "candidate_backend": args.candidate_backend,
            "bins": [asdict(move_bin) for move_bin in bins],
            "samples_per_bin": args.samples_per_bin,
            "seed": args.seed,
            "max_games": args.max_games,
            "games_seen": games_seen,
            "elapsed_seconds": time.time() - started,
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"policy_forward_single_step_ranking_{args.split}_{timestamp}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}_summary.json"
    md_path = out_dir / f"{stem}_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RankingRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summary)

    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")
    print(f"Markdown saved to: {md_path}")


if __name__ == "__main__":
    main()
