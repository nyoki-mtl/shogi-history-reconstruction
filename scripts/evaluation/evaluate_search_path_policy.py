#!/usr/bin/env python3
"""Measure move naturalness of widening-beam rollback paths with AlphaZero policy.

This script runs the ``iterative_widening_beam_rollback`` search (mode
``changed``) from a true game position back toward the initial position and
scores the resulting forward path with AlphaZero/dlshogi policy probabilities.
For each forward step ``(predecessor -> successor)`` the move USI is identified
with rsshogi and its probability is normalized by the legal-move probability mass
of the predecessor position.  The true game path of the same prefix is scored
the same way as a reference, so the model path and the human path are directly
comparable on per-move policy likelihood.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import geometric_mean, median
from typing import Any

import numpy as np
import torch
import webdataset as wds
from omegaconf import OmegaConf
from rsshogi.core import Board
from rsshogi.policy import move_label

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.evaluation.policy.alphazero_onnx import AlphaZeroONNX
from posdiffusion.inference import load_model
from posdiffusion.reverse.expansion import BLEND_MODE, ExpansionProvider, make_alphazero_policy_scorer
from posdiffusion.reverse.search import Tokens, iterative_widening_beam_rollback

SCORE_MODE_CHOICES = ("changed", BLEND_MODE)


@dataclass
class SearchPathPolicyRow:
    game_id: str
    start_move: int
    reached_initial: bool
    expansions_used: int
    steps_to_initial: int | None
    model_path_policy_mean: float | None
    model_path_policy_median: float | None
    model_path_policy_min: float | None
    model_path_policy_geomean: float | None
    true_path_policy_mean: float | None
    true_path_policy_median: float | None
    true_path_policy_min: float | None
    true_path_policy_geomean: float | None
    model_exact_step_count: int | None
    elapsed_seconds: float


def load_test_pattern_from_config(config_path: str | Path) -> str:
    cfg = OmegaConf.load(config_path)
    pattern = OmegaConf.select(cfg, "data.test_shard_pattern") or OmegaConf.select(cfg, "test_shard_pattern")
    if pattern is None:
        raise ValueError(f"Config {config_path} must contain data.test_shard_pattern")
    return str(pattern)


def sfen_core(board: Board) -> str:
    return " ".join(board.to_sfen().split()[:3])


def find_forward_move(predecessor: Board, successor: Board) -> str:
    """Return the USI move that turns ``predecessor`` into ``successor``."""
    target = sfen_core(successor)
    for move in predecessor.legal_moves():
        predecessor.push_move(move)
        matched = sfen_core(predecessor) == target
        predecessor.pop()
        if matched:
            return move.to_usi()
    raise ValueError("No legal forward move connects the two positions")


def path_policy_probs(
    forward_path: list[Tokens],
    *,
    tokenizer: ShogiTokenizer,
    alphazero: AlphaZeroONNX,
) -> list[float]:
    """Return per-step normalized policy probabilities along a forward path.

    ``forward_path`` is ordered initial -> start (each entry a 96-token tuple).
    For step ``(pred, succ)`` the predecessor board policy is evaluated and the
    forward move probability is normalized by the legal-move probability mass.
    """
    if len(forward_path) < 2:
        return []
    predecessor_boards = [tokenizer.decode(list(forward_path[index])) for index in range(len(forward_path) - 1)]
    successor_boards = [tokenizer.decode(list(forward_path[index])) for index in range(1, len(forward_path))]
    policy_probs_batch, _ = alphazero.get_policy_value_batch(predecessor_boards)

    probabilities: list[float] = []
    for predecessor_board, successor_board, policy_probs in zip(
        predecessor_boards, successor_boards, policy_probs_batch, strict=True
    ):
        move_usi = find_forward_move(predecessor_board, successor_board)
        move = next(m for m in predecessor_board.legal_moves() if m.to_usi() == move_usi)
        legal_total = 0.0
        for legal_move in predecessor_board.legal_moves():
            legal_total += float(policy_probs[move_label(legal_move, predecessor_board.turn)])
        probability = float(policy_probs[move_label(move, predecessor_board.turn)])
        if legal_total > 0.0:
            probability /= legal_total
        probabilities.append(probability)
    return probabilities


def summarize_probs(probs: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, median, min, geometric mean) of a non-empty list."""
    return (
        float(np.mean(probs)),
        float(median(probs)),
        float(min(probs)),
        float(geometric_mean(probs)),
    )


def count_exact_steps(model_forward_path: list[Tokens], true_forward_path: list[Tokens]) -> int:
    """Count steps whose ``(pred, succ)`` transition matches the true path."""
    true_steps = {
        (true_forward_path[index], true_forward_path[index + 1]) for index in range(len(true_forward_path) - 1)
    }
    return sum(
        (model_forward_path[index], model_forward_path[index + 1]) in true_steps
        for index in range(len(model_forward_path) - 1)
    )


def parse_int_list(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if any(value <= 0 for value in values):
        raise ValueError("All values must be positive")
    return values


def aggregate(rows: list[SearchPathPolicyRow], start_moves: list[int]) -> dict[str, Any]:
    by_start_move: dict[str, Any] = {}
    for start_move in start_moves:
        selected = [row for row in rows if row.start_move == start_move]
        if not selected:
            continue
        reached = [row for row in selected if row.reached_initial]
        model_means = [row.model_path_policy_mean for row in reached if row.model_path_policy_mean is not None]
        true_means_success = [row.true_path_policy_mean for row in reached if row.true_path_policy_mean is not None]
        true_means_all = [row.true_path_policy_mean for row in selected if row.true_path_policy_mean is not None]
        paired_diffs = [
            row.model_path_policy_mean - row.true_path_policy_mean
            for row in reached
            if row.model_path_policy_mean is not None and row.true_path_policy_mean is not None
        ]
        by_start_move[str(start_move)] = {
            "num_records": len(selected),
            "success_rate": len(reached) / len(selected),
            "model_policy_mean_avg": float(np.mean(model_means)) if model_means else None,
            "model_policy_mean_median": float(median(model_means)) if model_means else None,
            "true_policy_mean_avg_success": float(np.mean(true_means_success)) if true_means_success else None,
            "true_policy_mean_avg_all": float(np.mean(true_means_all)) if true_means_all else None,
            "model_minus_true_policy_mean_avg": float(np.mean(paired_diffs)) if paired_diffs else None,
        }
    return {"num_rows": len(rows), "by_start_move": by_start_move}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure policy naturalness of widening-beam rollback paths")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--alphazero_onnx", type=str, default="data/models/dl_suisho/model.onnx")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--beam_widths", type=str, default="1,3,10,30,100")
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--per_state_topk", type=int, default=10)
    parser.add_argument("--moves", type=str, default="40,50,60,80")
    parser.add_argument("--skip_games", "--game_start", dest="skip_games", type=int, default=0)
    parser.add_argument("--max_games", type=int, default=10, help="0 means no limit")
    parser.add_argument(
        "--min_positions",
        type=int,
        default=0,
        help="Games with fewer than this many positions are skipped without counting",
    )
    parser.add_argument("--candidate_backend", choices=["rsshogi", "yaneuraou", "yaneuraou_hybrid"], default="rsshogi")
    parser.add_argument("--score_mode", choices=list(SCORE_MODE_CHOICES), default="changed")
    parser.add_argument("--policy_alpha", type=float, default=1.0)
    parser.add_argument("--policy_prefilter_k", type=int, default=30)
    parser.add_argument("--policy_floor", type=float, default=1.0e-6)
    parser.add_argument(
        "--policy_h_threshold",
        type=int,
        default=None,
        help="Apply policy blending only when the distance-to-initial lower bound is at most this value",
    )
    parser.add_argument("--output_dir", type=str, default="work_dir/evaluation/search_path_policy")
    parser.add_argument("--progress_every_games", type=int, default=1)
    args = parser.parse_args()

    if args.skip_games < 0:
        raise ValueError("--skip_games must be non-negative")
    if args.progress_every_games <= 0:
        raise ValueError("--progress_every_games must be positive")
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.per_state_topk <= 0:
        raise ValueError("--per_state_topk must be positive")
    if args.policy_prefilter_k <= 0:
        raise ValueError("--policy_prefilter_k must be positive")
    if args.policy_floor <= 0.0:
        raise ValueError("--policy_floor must be positive")
    if args.min_positions < 0:
        raise ValueError("--min_positions must be non-negative")

    beam_widths = tuple(parse_int_list(args.beam_widths))
    start_moves = parse_int_list(args.moves)

    tokenizer = ShogiTokenizer(flip_to_black=False)
    model = load_model(args.checkpoint, args.device)
    alphazero = AlphaZeroONNX(args.alphazero_onnx, device="cpu")
    # The measurement and the blend scorer share the same AlphaZero instance.
    policy_scorer = make_alphazero_policy_scorer(alphazero, tokenizer) if args.score_mode == BLEND_MODE else None

    dataset = wds.WebDataset(load_test_pattern_from_config(args.config), shardshuffle=False).decode()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"search_path_policy_{timestamp}.csv"
    json_path = out_dir / f"search_path_policy_summary_{timestamp}.json"

    rows: list[SearchPathPolicyRow] = []
    game_count = 0
    started = time.time()
    fieldnames = list(SearchPathPolicyRow.__dataclass_fields__.keys())
    print(f"Writing incremental CSV to: {csv_path}", flush=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx, sample in enumerate(dataset):
            if sample_idx < args.skip_games:
                continue
            positions = sample["positions.npy"]
            if len(positions) < args.min_positions:
                continue
            game_id = str(sample.get("game_id.txt", sample.get("__key__", "unknown")))
            initial_tokens = tuple(int(token) for token in positions[0])
            game_rows: list[SearchPathPolicyRow] = []
            provider = ExpansionProvider(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                candidate_backend=args.candidate_backend,
                per_state_topk=args.per_state_topk,
                policy_scorer=policy_scorer,
                policy_alpha=args.policy_alpha,
                policy_prefilter_k=args.policy_prefilter_k,
                policy_floor=args.policy_floor,
                policy_h_threshold=args.policy_h_threshold,
                initial_tokens=initial_tokens,
            )
            expand = provider.expand_fn(args.score_mode)
            for start_move in start_moves:
                if start_move >= len(positions):
                    continue
                start_tokens = tuple(int(token) for token in positions[start_move])
                # True forward path: initial -> start position (96-token tuples).
                true_forward_path = [tuple(int(token) for token in positions[step]) for step in range(start_move + 1)]
                true_probs = path_policy_probs(true_forward_path, tokenizer=tokenizer, alphazero=alphazero)
                true_summary = summarize_probs(true_probs) if true_probs else None

                run_started = time.time()
                result = iterative_widening_beam_rollback(
                    start_tokens=start_tokens,
                    initial_tokens=initial_tokens,
                    expand_fn=expand,
                    max_steps=start_move,
                    expansion_budget=args.budget,
                    widths=beam_widths,
                )
                model_summary: tuple[float, float, float, float] | None = None
                exact_step_count: int | None = None
                if result.reached_initial and result.path is not None:
                    # result.path is [start, ..., initial]; reverse to initial -> start.
                    model_forward_path = list(reversed(result.path))
                    model_probs = path_policy_probs(model_forward_path, tokenizer=tokenizer, alphazero=alphazero)
                    if model_probs:
                        model_summary = summarize_probs(model_probs)
                    exact_step_count = count_exact_steps(model_forward_path, true_forward_path)

                game_rows.append(
                    SearchPathPolicyRow(
                        game_id=game_id,
                        start_move=start_move,
                        reached_initial=result.reached_initial,
                        expansions_used=result.expansions,
                        steps_to_initial=result.steps_to_initial,
                        model_path_policy_mean=model_summary[0] if model_summary else None,
                        model_path_policy_median=model_summary[1] if model_summary else None,
                        model_path_policy_min=model_summary[2] if model_summary else None,
                        model_path_policy_geomean=model_summary[3] if model_summary else None,
                        true_path_policy_mean=true_summary[0] if true_summary else None,
                        true_path_policy_median=true_summary[1] if true_summary else None,
                        true_path_policy_min=true_summary[2] if true_summary else None,
                        true_path_policy_geomean=true_summary[3] if true_summary else None,
                        model_exact_step_count=exact_step_count,
                        elapsed_seconds=time.time() - run_started,
                    )
                )

            rows.extend(game_rows)
            for row in game_rows:
                writer.writerow(asdict(row))
            f.flush()

            game_count += 1
            if game_count % args.progress_every_games == 0:
                elapsed = time.time() - started
                print(
                    f"Progress: games={game_count} rows={len(rows)} "
                    f"elapsed={elapsed:.1f}s sec_per_game={elapsed / game_count:.2f} last_game_id={game_id}",
                    flush=True,
                )
            if args.max_games > 0 and game_count >= args.max_games:
                break

    summary = aggregate(rows, start_moves)
    summary.update(
        {
            "num_games": game_count,
            "skip_games": args.skip_games,
            "max_games": args.max_games,
            "min_positions": args.min_positions,
            "beam_widths": list(beam_widths),
            "budget": args.budget,
            "per_state_topk": args.per_state_topk,
            "moves": start_moves,
            "candidate_backend": args.candidate_backend,
            "score_mode": args.score_mode,
            "policy_alpha": args.policy_alpha,
            "policy_prefilter_k": args.policy_prefilter_k,
            "policy_floor": args.policy_floor,
            "policy_h_threshold": args.policy_h_threshold,
            "checkpoint": args.checkpoint,
            "alphazero_onnx": args.alphazero_onnx,
            "elapsed_seconds": time.time() - started,
            "csv_path": str(csv_path),
        }
    )
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed games: {game_count}")
    print(f"Rows: {len(rows)}")
    print(f"Summary saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
