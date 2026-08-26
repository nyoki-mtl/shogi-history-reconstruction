#!/usr/bin/env python3
"""Evaluate legal-projected history reconstruction.

The available algorithms use an admissible distance-to-initial lower bound to
prune branches that cannot reach the initial position within the step limit.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import webdataset as wds
from omegaconf import OmegaConf

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.evaluation.policy.alphazero_onnx import AlphaZeroONNX
from posdiffusion.inference import load_model
from posdiffusion.reverse.expansion import (
    BLEND_MODE,
    CONTROL_MODES,
    MODEL_MODES,
    POLICY_MODE,
    ExpansionProvider,
    make_alphazero_policy_scorer,
)
from posdiffusion.reverse.search import best_first_rollback, iterative_widening_beam_rollback


@dataclass
class SearchRollbackRow:
    game_id: str
    start_move: int
    mode: str
    algorithm: str
    expansion_budget: int
    step_cost: float
    step_bonus: float
    per_state_topk: int
    max_steps: int
    reached_initial: bool
    steps_to_initial: int | None
    exact_path: bool
    true_state_overlap: float | None
    expansions_used: int
    generated: int
    stop_reason: str
    start_lower_bound: int
    max_depth_expanded: int
    min_bound_seen: int
    elapsed_seconds: float


def load_test_pattern_from_config(config_path: str | Path) -> str:
    cfg = OmegaConf.load(config_path)
    pattern = OmegaConf.select(cfg, "data.test_shard_pattern") or OmegaConf.select(cfg, "test_shard_pattern")
    if pattern is None:
        raise ValueError(f"Config {config_path} must contain data.test_shard_pattern")
    return str(pattern)


def parse_float_list(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if any(value <= 0 for value in values):
        raise ValueError("All values must be positive")
    return values


def aggregate(rows: list[SearchRollbackRow]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_rows": len(rows)}
    by_setting: dict[str, dict[str, Any]] = {}
    keys = sorted({(row.start_move, row.mode, row.expansion_budget, row.step_cost, row.step_bonus) for row in rows})
    for start_move, mode, budget, step_cost, step_bonus in keys:
        selected = [
            row
            for row in rows
            if row.start_move == start_move
            and row.mode == mode
            and row.expansion_budget == budget
            and row.step_cost == step_cost
            and row.step_bonus == step_bonus
        ]
        key = f"move{start_move}/{mode}/budget{budget}/cost{step_cost:g}/bonus{step_bonus:g}"
        reached = [row for row in selected if row.reached_initial]
        by_setting[key] = {
            "num_records": len(selected),
            "reached_rate": len(reached) / len(selected),
            "exact_path_rate": sum(row.exact_path for row in selected) / len(selected),
            "avg_expansions": float(np.mean([row.expansions_used for row in selected])),
            "avg_expansions_success": float(np.mean([row.expansions_used for row in reached])) if reached else None,
            "avg_steps_to_initial": float(np.mean([row.steps_to_initial for row in reached])) if reached else None,
            "avg_true_state_overlap_success": float(np.mean([row.true_state_overlap for row in reached]))
            if reached
            else None,
            "avg_elapsed_seconds": float(np.mean([row.elapsed_seconds for row in selected])),
            "stop_reasons": {
                reason: sum(1 for row in selected if row.stop_reason == reason)
                for reason in sorted({row.stop_reason for row in selected})
            },
        }
    summary["by_setting"] = by_setting
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best-first projection rollback")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--alphazero_onnx",
        type=str,
        help="Required when mode changed_policy_blend or policy_forward is selected",
    )
    parser.add_argument("--policy_alpha", type=float, default=1.0)
    parser.add_argument("--policy_prefilter_k", type=int, default=30)
    parser.add_argument("--policy_floor", type=float, default=1.0e-6)
    parser.add_argument(
        "--policy_h_threshold",
        type=int,
        default=None,
        help="Apply policy blending only when the distance-to-initial lower bound is at most this value",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--modes", type=str, default="changed")
    parser.add_argument("--budgets", type=str, default="500", help="Comma-separated expansion budgets")
    parser.add_argument("--step_costs", type=str, default="1.0", help="Comma-separated heuristic weights")
    parser.add_argument("--step_bonuses", type=str, default="0.0", help="Comma-separated depth-reward weights")
    parser.add_argument("--algorithm", choices=["best_first", "widening_beam"], default="best_first")
    parser.add_argument("--beam_widths", type=str, default="1,3,10,30,100", help="Widths for widening_beam")
    parser.add_argument("--per_state_topk", type=int, default=10)
    parser.add_argument("--moves", type=str, default="40,50")
    parser.add_argument(
        "--extra_rollback_steps",
        type=int,
        default=0,
        help="Depth-bound slack beyond each start move",
    )
    parser.add_argument("--skip_games", "--game_start", dest="skip_games", type=int, default=0)
    parser.add_argument("--min_positions", type=int, default=0, help="Skip games with fewer positions than this")
    parser.add_argument("--max_games", type=int, default=10, help="0 means no limit")
    parser.add_argument("--game_limit", dest="max_games", type=int)
    parser.add_argument("--output_dir", type=str, default="work_dir/evaluation/projection_search_rollback")
    parser.add_argument("--progress_every_games", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--candidate_backend", choices=["rsshogi", "yaneuraou", "yaneuraou_hybrid"], default="rsshogi")
    args = parser.parse_args()

    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    unknown = sorted(set(modes) - MODEL_MODES - CONTROL_MODES - {BLEND_MODE, POLICY_MODE})
    if unknown:
        raise ValueError(f"Unknown score modes: {unknown}")
    if args.policy_prefilter_k <= 0:
        raise ValueError("--policy_prefilter_k must be positive")
    if args.policy_floor <= 0.0:
        raise ValueError("--policy_floor must be positive")
    budgets = parse_int_list(args.budgets)
    step_costs = parse_float_list(args.step_costs)
    step_bonuses = parse_float_list(args.step_bonuses)
    beam_widths = tuple(parse_int_list(args.beam_widths))
    if args.algorithm == "widening_beam":
        # Width schedule replaces the priority parameters; report them as zero.
        step_costs = [0.0]
        step_bonuses = [0.0]
    start_moves = parse_int_list(args.moves)
    if args.skip_games < 0:
        raise ValueError("--skip_games must be non-negative")
    if args.min_positions < 0:
        raise ValueError("--min_positions must be non-negative")
    if args.extra_rollback_steps < 0:
        raise ValueError("--extra_rollback_steps must be non-negative")

    tokenizer = ShogiTokenizer(flip_to_black=False)
    if any(mode in MODEL_MODES or mode == BLEND_MODE for mode in modes):
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when any selected mode uses model scores")
        model = load_model(args.checkpoint, args.device)
    else:
        model = None
    if BLEND_MODE in modes or POLICY_MODE in modes:
        if args.alphazero_onnx is None:
            raise ValueError("--alphazero_onnx is required for policy-based score modes")
        policy_scorer = make_alphazero_policy_scorer(AlphaZeroONNX(args.alphazero_onnx, device="cpu"), tokenizer)
    else:
        policy_scorer = None

    dataset = wds.WebDataset(load_test_pattern_from_config(args.config), shardshuffle=False).decode()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"projection_search_rollback_{timestamp}.csv"
    json_path = out_dir / f"projection_search_rollback_summary_{timestamp}.json"

    rows: list[SearchRollbackRow] = []
    game_count = 0
    started = time.time()
    fieldnames = list(SearchRollbackRow.__dataclass_fields__.keys())
    print(f"Writing incremental CSV to: {csv_path}", flush=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx, sample in enumerate(dataset):
            if sample_idx < args.skip_games:
                continue
            game_id = str(sample.get("game_id.txt", sample.get("__key__", "unknown")))
            positions = sample["positions.npy"]
            if args.min_positions > 0 and len(positions) < args.min_positions:
                continue
            initial_tokens = tuple(int(token) for token in positions[0])
            game_rows: list[SearchRollbackRow] = []
            for start_move in start_moves:
                if start_move >= len(positions):
                    continue
                start_tokens = tuple(int(token) for token in positions[start_move])
                true_path = [
                    tuple(int(token) for token in positions[start_move - step]) for step in range(start_move + 1)
                ]
                provider = ExpansionProvider(
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    candidate_backend=args.candidate_backend,
                    per_state_topk=args.per_state_topk,
                    random_seed=args.random_seed,
                    policy_scorer=policy_scorer,
                    policy_alpha=args.policy_alpha,
                    policy_prefilter_k=args.policy_prefilter_k,
                    policy_floor=args.policy_floor,
                    policy_h_threshold=args.policy_h_threshold,
                    initial_tokens=initial_tokens,
                )
                for mode in modes:
                    expand = provider.expand_fn(mode)
                    for budget in budgets:
                        for step_cost in step_costs:
                            for step_bonus in step_bonuses:
                                run_started = time.time()
                                if args.algorithm == "widening_beam":
                                    result = iterative_widening_beam_rollback(
                                        start_tokens=start_tokens,
                                        initial_tokens=initial_tokens,
                                        expand_fn=expand,
                                        max_steps=start_move + args.extra_rollback_steps,
                                        expansion_budget=budget,
                                        widths=beam_widths,
                                    )
                                else:
                                    result = best_first_rollback(
                                        start_tokens=start_tokens,
                                        initial_tokens=initial_tokens,
                                        expand_fn=expand,
                                        max_steps=start_move + args.extra_rollback_steps,
                                        expansion_budget=budget,
                                        step_cost=step_cost,
                                        step_bonus=step_bonus,
                                    )
                                exact_path = bool(result.reached_initial and result.path == true_path)
                                if result.path is not None:
                                    true_states = set(true_path)
                                    overlap = sum(state in true_states for state in result.path) / len(result.path)
                                else:
                                    overlap = None
                                game_rows.append(
                                    SearchRollbackRow(
                                        game_id=game_id,
                                        start_move=start_move,
                                        mode=mode,
                                        algorithm=args.algorithm,
                                        expansion_budget=budget,
                                        step_cost=step_cost,
                                        step_bonus=step_bonus,
                                        per_state_topk=args.per_state_topk,
                                        max_steps=start_move + args.extra_rollback_steps,
                                        reached_initial=result.reached_initial,
                                        steps_to_initial=result.steps_to_initial,
                                        exact_path=exact_path,
                                        true_state_overlap=overlap,
                                        expansions_used=result.expansions,
                                        generated=result.generated,
                                        stop_reason=result.stop_reason,
                                        start_lower_bound=result.start_lower_bound,
                                        max_depth_expanded=result.max_depth_expanded,
                                        min_bound_seen=result.min_bound_seen,
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

    summary = aggregate(rows)
    summary.update(
        {
            "num_games": game_count,
            "skip_games": args.skip_games,
            "min_positions": args.min_positions,
            "max_games": args.max_games,
            "modes": modes,
            "budgets": budgets,
            "step_costs": step_costs,
            "step_bonuses": step_bonuses,
            "per_state_topk": args.per_state_topk,
            "extra_rollback_steps": args.extra_rollback_steps,
            "moves": start_moves,
            "random_seed": args.random_seed,
            "candidate_backend": args.candidate_backend,
            "checkpoint": args.checkpoint,
            "alphazero_onnx": args.alphazero_onnx,
            "policy_alpha": args.policy_alpha,
            "policy_prefilter_k": args.policy_prefilter_k,
            "policy_floor": args.policy_floor,
            "policy_h_threshold": args.policy_h_threshold,
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
