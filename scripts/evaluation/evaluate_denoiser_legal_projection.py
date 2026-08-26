#!/usr/bin/env python3
"""Evaluate denoiser legal projection over generated predecessor candidates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import webdataset as wds
from omegaconf import OmegaConf

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.diffusion.transformer_denoiser import ShogiTransformerDenoiser
from posdiffusion.inference import load_model
from posdiffusion.reverse.candidates import CandidateBackend, generate_predecessor_candidates
from posdiffusion.reverse.scoring import (
    ScoreMode,
    candidates_to_tensor,
    score_candidate_tokens,
    true_candidate_index,
)


@dataclass
class ProjectionRow:
    game_id: str
    move_index: int
    mode: str
    candidate_count: int
    true_predecessor_covered: bool
    true_rank: int | None
    top1_exact: bool
    top3_exact: bool
    top5_exact: bool
    selected_move_usi: str | None
    true_score: float | None
    best_score: float | None


@dataclass
class BaselineRow:
    game_id: str
    move_index: int
    baseline: str
    candidate_count: int
    true_predecessor_covered: bool
    top1_exact: bool
    selected_move_usi: str | None
    selected_is_reverse_legal: bool


def load_shard_pattern(config_path: str | Path, split: str) -> str:
    cfg = OmegaConf.load(config_path)
    pattern = OmegaConf.select(cfg, f"data.{split}_shard_pattern") or OmegaConf.select(cfg, f"{split}_shard_pattern")
    if pattern is None:
        raise ValueError(f"Config {config_path} must contain data.{split}_shard_pattern")
    return str(pattern)


def _rank_of_true(scores: torch.Tensor, true_idx: int) -> int:
    true_score = scores[true_idx]
    return int((scores > true_score).sum().item()) + 1


def _true_in_topk(scores: torch.Tensor, true_idx: int, k: int) -> bool:
    topk = torch.topk(scores, k=min(k, scores.numel())).indices
    return bool((topk == true_idx).any().item())


def _token_change_counts(source_tokens: torch.Tensor, candidate_tokens: torch.Tensor) -> torch.Tensor:
    return (candidate_tokens != source_tokens.unsqueeze(0)).to(torch.int64).sum(dim=1)


def evaluate_game(
    *,
    model: ShogiTransformerDenoiser,
    tokenizer: ShogiTokenizer,
    device: str,
    game_id: str,
    positions: np.ndarray,
    modes: list[ScoreMode],
    max_positions: int,
    candidate_backend: CandidateBackend,
) -> tuple[list[ProjectionRow], list[BaselineRow]]:
    rows: list[ProjectionRow] = []
    baseline_rows: list[BaselineRow] = []
    end = len(positions)
    if max_positions > 0:
        end = min(end, max_positions + 1)

    for move_idx in range(1, end):
        true_tokens = tuple(int(token) for token in positions[move_idx - 1])
        source_np = positions[move_idx]
        candidates = generate_predecessor_candidates(source_np, tokenizer=tokenizer, backend=candidate_backend)
        true_idx = true_candidate_index(candidates, true_tokens)
        covered = true_idx is not None
        candidate_count = len(candidates)
        candidate_token_set = {candidate.tokens for candidate in candidates}

        if candidate_count == 0:
            for mode in modes:
                rows.append(
                    ProjectionRow(
                        game_id=game_id,
                        move_index=move_idx,
                        mode=mode,
                        candidate_count=0,
                        true_predecessor_covered=False,
                        true_rank=None,
                        top1_exact=False,
                        top3_exact=False,
                        top5_exact=False,
                        selected_move_usi=None,
                        true_score=None,
                        best_score=None,
                    )
                )
            baseline_rows.extend(
                [
                    BaselineRow(
                        game_id=game_id,
                        move_index=move_idx,
                        baseline=baseline,
                        candidate_count=0,
                        true_predecessor_covered=False,
                        top1_exact=False,
                        selected_move_usi=None,
                        selected_is_reverse_legal=False,
                    )
                    for baseline in ("token_argmax", "first_legal_rule_order", "min_token_change_legal")
                ]
            )
            continue

        source_tokens = torch.tensor(source_np, dtype=torch.long, device=device)
        candidate_tokens = candidates_to_tensor(candidates, device=device)
        with torch.no_grad():
            logits = model(
                source_tokens.unsqueeze(0),
                torch.ones(1, device=device, dtype=torch.long),
                absolute_move=None,
                game_phase=None,
            )[0]

        argmax_tokens = tuple(int(token) for token in torch.argmax(logits, dim=-1).detach().cpu().tolist())
        baseline_rows.append(
            BaselineRow(
                game_id=game_id,
                move_index=move_idx,
                baseline="token_argmax",
                candidate_count=candidate_count,
                true_predecessor_covered=covered,
                top1_exact=argmax_tokens == true_tokens,
                selected_move_usi=None,
                selected_is_reverse_legal=argmax_tokens in candidate_token_set,
            )
        )
        baseline_rows.append(
            BaselineRow(
                game_id=game_id,
                move_index=move_idx,
                baseline="first_legal_rule_order",
                candidate_count=candidate_count,
                true_predecessor_covered=covered,
                top1_exact=true_idx == 0,
                selected_move_usi=candidates[0].move_usi,
                selected_is_reverse_legal=True,
            )
        )
        min_change_idx = int(torch.argmin(_token_change_counts(source_tokens, candidate_tokens)).item())
        baseline_rows.append(
            BaselineRow(
                game_id=game_id,
                move_index=move_idx,
                baseline="min_token_change_legal",
                candidate_count=candidate_count,
                true_predecessor_covered=covered,
                top1_exact=true_idx == min_change_idx,
                selected_move_usi=candidates[min_change_idx].move_usi,
                selected_is_reverse_legal=True,
            )
        )

        for mode in modes:
            scores = score_candidate_tokens(logits, source_tokens, candidate_tokens, mode=mode).scores
            best_idx = int(torch.argmax(scores).item())
            true_rank = _rank_of_true(scores, true_idx) if true_idx is not None else None
            top1_exact = true_idx is not None and best_idx == true_idx
            top3_exact = true_idx is not None and _true_in_topk(scores, true_idx, 3)
            top5_exact = true_idx is not None and _true_in_topk(scores, true_idx, 5)
            rows.append(
                ProjectionRow(
                    game_id=game_id,
                    move_index=move_idx,
                    mode=mode,
                    candidate_count=candidate_count,
                    true_predecessor_covered=covered,
                    true_rank=true_rank,
                    top1_exact=top1_exact,
                    top3_exact=top3_exact,
                    top5_exact=top5_exact,
                    selected_move_usi=candidates[best_idx].move_usi,
                    true_score=float(scores[true_idx].item()) if true_idx is not None else None,
                    best_score=float(scores[best_idx].item()),
                )
            )
    return rows, baseline_rows


def aggregate(rows: list[ProjectionRow], baseline_rows: list[BaselineRow]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_rows": len(rows)}
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({row.mode for row in rows}):
        mode_rows = [row for row in rows if row.mode == mode]
        covered_rows = [row for row in mode_rows if row.true_predecessor_covered]
        candidate_counts = np.asarray([row.candidate_count for row in mode_rows], dtype=np.int64)
        random_top1_terms = [1.0 / row.candidate_count for row in covered_rows if row.candidate_count > 0]
        true_scores = [row.true_score for row in covered_rows if row.true_score is not None]
        by_mode[mode] = {
            "num_positions": len(mode_rows),
            "true_predecessor_coverage": float(len(covered_rows) / len(mode_rows)) if mode_rows else 0.0,
            "top1_exact": float(sum(row.top1_exact for row in mode_rows) / len(mode_rows)) if mode_rows else 0.0,
            "top3_exact": float(sum(row.top3_exact for row in mode_rows) / len(mode_rows)) if mode_rows else 0.0,
            "top5_exact": float(sum(row.top5_exact for row in mode_rows) / len(mode_rows)) if mode_rows else 0.0,
            "covered_top1_exact": float(sum(row.top1_exact for row in covered_rows) / len(covered_rows))
            if covered_rows
            else 0.0,
            "covered_top3_exact": float(sum(row.top3_exact for row in covered_rows) / len(covered_rows))
            if covered_rows
            else 0.0,
            "covered_top5_exact": float(sum(row.top5_exact for row in covered_rows) / len(covered_rows))
            if covered_rows
            else 0.0,
            "random_legal_top1_expected": float(np.mean(random_top1_terms)) if random_top1_terms else 0.0,
            "true_nll_mean": float(np.mean([-score for score in true_scores])) if true_scores else None,
            "true_nll_median": float(np.median([-score for score in true_scores])) if true_scores else None,
            "candidate_count_mean": float(candidate_counts.mean()) if len(candidate_counts) else 0.0,
            "candidate_count_median": float(np.median(candidate_counts)) if len(candidate_counts) else 0.0,
            "candidate_count_p90": float(np.quantile(candidate_counts, 0.9)) if len(candidate_counts) else 0.0,
            "candidate_count_max": int(candidate_counts.max()) if len(candidate_counts) else 0,
        }
    summary["by_mode"] = by_mode
    by_baseline: dict[str, dict[str, Any]] = {}
    for baseline in sorted({row.baseline for row in baseline_rows}):
        selected_rows = [row for row in baseline_rows if row.baseline == baseline]
        covered_rows = [row for row in selected_rows if row.true_predecessor_covered]
        legal_rows = [row for row in selected_rows if row.selected_is_reverse_legal]
        by_baseline[baseline] = {
            "num_positions": len(selected_rows),
            "true_predecessor_coverage": float(len(covered_rows) / len(selected_rows)) if selected_rows else 0.0,
            "top1_exact": float(sum(row.top1_exact for row in selected_rows) / len(selected_rows))
            if selected_rows
            else 0.0,
            "covered_top1_exact": float(sum(row.top1_exact for row in covered_rows) / len(covered_rows))
            if covered_rows
            else 0.0,
            "selected_reverse_legal_rate": float(len(legal_rows) / len(selected_rows)) if selected_rows else 0.0,
        }
    summary["baselines"] = by_baseline
    return summary


def write_progress(
    *,
    path: Path,
    started_at: float,
    num_games_completed: int,
    rows_written: int,
    baseline_rows_written: int,
    split: str,
    skip_games: int,
    max_games: int,
    max_positions_per_game: int,
    modes: list[ScoreMode],
    candidate_backend: CandidateBackend,
    last_game_id: str | None,
    csv_path: Path,
    baseline_csv_path: Path,
) -> None:
    now = time.time()
    elapsed = now - started_at
    payload = {
        "started_at_unix": started_at,
        "updated_at_unix": now,
        "elapsed_seconds": elapsed,
        "num_games_completed": num_games_completed,
        "rows_written": rows_written,
        "baseline_rows_written": baseline_rows_written,
        "split": split,
        "skip_games": skip_games,
        "max_games": max_games,
        "max_positions_per_game": max_positions_per_game,
        "modes": modes,
        "candidate_backend": candidate_backend,
        "last_game_id": last_game_id,
        "csv_path": str(csv_path),
        "baseline_csv_path": str(baseline_csv_path),
        "seconds_per_game": elapsed / num_games_completed if num_games_completed else None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_modes(raw: str) -> list[ScoreMode]:
    modes = [mode.strip() for mode in raw.split(",") if mode.strip()]
    allowed = {"full", "changed", "board", "hand", "turn", "board_changed", "hand_changed"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown score modes: {unknown}")
    return cast(list[ScoreMode], list(modes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate denoiser legal projection")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--modes", type=str, default="full,changed,board,hand,turn,board_changed,hand_changed")
    parser.add_argument("--skip_games", "--game_start", dest="skip_games", type=int, default=0)
    parser.add_argument("--max_games", type=int, default=10, help="0 means no limit")
    parser.add_argument("--game_limit", dest="max_games", type=int)
    parser.add_argument("--max_positions_per_game", type=int, default=20, help="0 means all")
    parser.add_argument("--csv_limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--output_dir", type=str, default="work_dir/evaluation/denoiser_legal_projection")
    parser.add_argument("--candidate_backend", choices=["rsshogi", "yaneuraou", "yaneuraou_hybrid"], default="rsshogi")
    parser.add_argument("--progress_every_games", type=int, default=1)
    args = parser.parse_args()
    if args.skip_games < 0:
        raise ValueError("--skip_games must be non-negative")
    if args.progress_every_games <= 0:
        raise ValueError("--progress_every_games must be positive")

    modes = parse_modes(args.modes)
    tokenizer = ShogiTokenizer(flip_to_black=False)
    model = load_model(args.checkpoint, args.device)
    shard_pattern = load_shard_pattern(args.config, args.split)
    dataset = wds.WebDataset(shard_pattern, shardshuffle=False).decode()

    rows: list[ProjectionRow] = []
    baseline_rows: list[BaselineRow] = []
    game_count = 0
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"denoiser_legal_projection_{args.split}_{timestamp}.csv"
    baseline_csv_path = out_dir / f"denoiser_legal_projection_baselines_{args.split}_{timestamp}.csv"
    json_path = out_dir / f"denoiser_legal_projection_{args.split}_summary_{timestamp}.json"
    progress_path = out_dir / f"denoiser_legal_projection_{args.split}_progress_{timestamp}.json"

    limit = args.csv_limit if args.csv_limit > 0 else None
    written_rows = 0
    written_baseline_rows = 0
    last_game_id: str | None = None
    print(f"Writing incremental CSV to: {csv_path}", flush=True)
    print(f"Writing baseline CSV to: {baseline_csv_path}", flush=True)
    print(f"Writing progress JSON to: {progress_path}", flush=True)
    with (
        csv_path.open("w", newline="", encoding="utf-8") as f,
        baseline_csv_path.open("w", newline="", encoding="utf-8") as baseline_f,
    ):
        writer = csv.DictWriter(f, fieldnames=list(ProjectionRow.__dataclass_fields__.keys()))
        baseline_writer = csv.DictWriter(baseline_f, fieldnames=list(BaselineRow.__dataclass_fields__.keys()))
        writer.writeheader()
        baseline_writer.writeheader()
        write_progress(
            path=progress_path,
            started_at=started,
            num_games_completed=game_count,
            rows_written=written_rows,
            baseline_rows_written=written_baseline_rows,
            split=args.split,
            skip_games=args.skip_games,
            max_games=args.max_games,
            max_positions_per_game=args.max_positions_per_game,
            modes=modes,
            candidate_backend=args.candidate_backend,
            last_game_id=last_game_id,
            csv_path=csv_path,
            baseline_csv_path=baseline_csv_path,
        )
        for sample_idx, sample in enumerate(dataset):
            if sample_idx < args.skip_games:
                continue
            game_id = str(sample.get("game_id.txt", sample.get("__key__", "unknown")))
            game_rows, game_baseline_rows = evaluate_game(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                game_id=game_id,
                positions=sample["positions.npy"],
                modes=modes,
                max_positions=args.max_positions_per_game,
                candidate_backend=args.candidate_backend,
            )
            rows.extend(game_rows)
            baseline_rows.extend(game_baseline_rows)
            for row in game_rows:
                if limit is None or written_rows < limit:
                    writer.writerow(asdict(row))
                    written_rows += 1
            for row in game_baseline_rows:
                if limit is None or written_baseline_rows < limit:
                    baseline_writer.writerow(asdict(row))
                    written_baseline_rows += 1
            f.flush()
            baseline_f.flush()
            game_count += 1
            last_game_id = game_id
            if game_count % args.progress_every_games == 0:
                elapsed = time.time() - started
                print(
                    "Progress: "
                    f"games={game_count} rows={len(rows)} baseline_rows={len(baseline_rows)} "
                    f"elapsed={elapsed:.1f}s sec_per_game={elapsed / game_count:.2f} "
                    f"last_game_id={last_game_id}",
                    flush=True,
                )
                write_progress(
                    path=progress_path,
                    started_at=started,
                    num_games_completed=game_count,
                    rows_written=written_rows,
                    baseline_rows_written=written_baseline_rows,
                    split=args.split,
                    skip_games=args.skip_games,
                    max_games=args.max_games,
                    max_positions_per_game=args.max_positions_per_game,
                    modes=modes,
                    candidate_backend=args.candidate_backend,
                    last_game_id=last_game_id,
                    csv_path=csv_path,
                    baseline_csv_path=baseline_csv_path,
                )
            if args.max_games > 0 and game_count >= args.max_games:
                break

    summary = aggregate(rows, baseline_rows)
    summary.update(
        {
            "split": args.split,
            "shard_pattern": shard_pattern,
            "checkpoint": args.checkpoint,
            "modes": modes,
            "num_games": game_count,
            "skip_games": args.skip_games,
            "max_games": args.max_games,
            "max_positions_per_game": args.max_positions_per_game,
            "candidate_backend": args.candidate_backend,
            "elapsed_seconds": time.time() - started,
            "progress_path": str(progress_path),
            "csv_path": str(csv_path),
            "baseline_csv_path": str(baseline_csv_path),
        }
    )

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed games: {game_count}")
    print(f"Rows: {len(rows)}")
    print(f"Summary saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Baseline CSV saved to: {baseline_csv_path}")


if __name__ == "__main__":
    main()
