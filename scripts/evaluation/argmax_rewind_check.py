"""Check if repeated argmax distance=1 can rewind to the initial position.

For each test game, take moves at 10/20/…/100 (within bounds), and iterate:
  current = position at N
  predict prev via argmax (distance=1)
  set current = predicted prev
until reaching move 0 (N steps) or failure limit.

Metrics per starting N:
  - steps_attempted
  - steps_success (how many steps we could step back with decode success)
  - reached_initial (whether final board equals initial position)
  - any_illegal_pred (whether any predicted prev evaluated as illegal by PositionValidator)

Outputs: CSV per-game and JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import webdataset as wds
from omegaconf import OmegaConf
from rsshogi.core import Board
from rsshogi.types import Color, PieceType, Square

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.board.validator import PositionValidator
from posdiffusion.diffusion.transformer_denoiser import ShogiTransformerDenoiser
from posdiffusion.inference import load_model

try:  # Best-effort progress bar; fall back to plain iterator if unavailable
    from tqdm.auto import tqdm as _tqdm

    TQDM_AVAILABLE = True
except Exception:  # pragma: no cover
    _tqdm = None  # type: ignore
    TQDM_AVAILABLE = False


def maybe_tqdm[T](iterable: Iterable[T], **kwargs: Any) -> Iterable[T] | Any:
    if TQDM_AVAILABLE and _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return iterable


HAND_PIECE_TYPES = [
    PieceType.PAWN,
    PieceType.LANCE,
    PieceType.KNIGHT,
    PieceType.SILVER,
    PieceType.GOLD,
    PieceType.BISHOP,
    PieceType.ROOK,
]


def _piece(board: Board, square: int) -> int:
    return int(board.piece_on(Square.from_index(square)))


def _hand(board: Board, color: Color) -> list[int]:
    return [board.hand(color).count(piece_type) for piece_type in HAND_PIECE_TYPES]


@dataclass
class RewindRow:
    game_id: str
    start_move: int
    steps_attempted: int
    steps_success: int
    reached_initial: bool
    any_illegal_pred: bool
    illegal_count: int
    first_illegal_step: int | None
    stop_reason: str
    final_sfen: str | None


def load_test_pattern_from_config(config_path: str | Path) -> str:
    cfg = OmegaConf.load(config_path)
    pattern = OmegaConf.select(cfg, "data.test_shard_pattern") or OmegaConf.select(cfg, "test_shard_pattern")
    if pattern is None:
        raise ValueError(f"Config {config_path} must contain data.test_shard_pattern")
    return str(pattern)


def boards_equal(a: Board, b: Board) -> bool:
    if a.turn != b.turn:
        return False
    for sq in range(81):
        if _piece(a, sq) != _piece(b, sq):
            return False
    for color in (Color.BLACK, Color.WHITE):
        if _hand(a, color) != _hand(b, color):
            return False
    return True


def safe_sfen(board: Board | None) -> str | None:
    if board is None:
        return None
    try:
        return board.to_sfen()
    except Exception as e:  # pragma: no cover
        return f"ERROR:{e}"


def count_total_pieces(board: Board) -> int:
    on_board = sum(1 for sq in range(81) if _piece(board, sq) != 0)
    in_hands = 0
    for color in (Color.BLACK, Color.WHITE):
        in_hands += sum(_hand(board, color))
    return on_board + in_hands


def has_double_pawn(board: Board) -> bool:
    for color in (Color.BLACK, Color.WHITE):
        for file_idx in range(9):
            count = 0
            for rank_idx in range(9):
                sq = file_idx * 9 + rank_idx
                piece = _piece(board, sq)
                if (color == Color.BLACK and piece == 1) or (color == Color.WHITE and piece == 17):
                    count += 1
            if count >= 2:
                return True
    return False


def rewind_from(
    model: ShogiTransformerDenoiser,
    tokenizer: ShogiTokenizer,
    validator: PositionValidator,
    device: str,
    game_id: str,
    positions: np.ndarray,
    start_move: int,
    reverse_backend: str,
    returnable_use_light_checks: bool,
) -> tuple[RewindRow, list[dict[str, Any]]]:
    # Initial board for comparison
    initial_board = tokenizer.decode(positions[0].tolist())

    steps_attempted = start_move
    steps_success = 0
    any_illegal = False
    illegal_count = 0
    first_illegal_step: int | None = None
    step_logs: list[dict[str, Any]] = []
    stop_reason = "exhausted_steps"

    # Current tokens start at N
    current_tokens = torch.tensor(positions[start_move], device=device, dtype=torch.long)

    for step_idx in range(1, start_move + 1):
        # For logging, decode current
        board_current: Board | None
        try:
            board_current = tokenizer.decode(current_tokens.detach().cpu().tolist())
            current_sfen = safe_sfen(board_current)
        except Exception:
            board_current = None
            current_sfen = None
        with torch.no_grad():
            logits = model(
                current_tokens.unsqueeze(0),
                torch.ones(1, device=device, dtype=torch.long),
                absolute_move=None,
                game_phase=None,
            )[0]
            argmax_tokens = torch.argmax(logits, dim=-1)

        # Decode predicted previous
        try:
            pred_prev_board = tokenizer.decode(argmax_tokens.detach().cpu().tolist())
        except Exception:
            step_logs.append(
                {
                    "game_id": game_id,
                    "start_move": start_move,
                    "step": step_idx,
                    "decode_error": True,
                    "current_sfen": current_sfen,
                }
            )
            stop_reason = "decode_error"
            break

        # Compute per-step stats and legal check
        try:
            change_count = int((current_tokens[:95] != argmax_tokens[:95]).to(torch.int).sum().item())
        except Exception:
            change_count = -1

        # Structural validity + existence of a legal move from pred_prev -> current
        def _exists_forward_legal_move(src: Board, dst: Board) -> bool:
            for mv in src.legal_moves():
                tmp = Board(src.to_sfen())
                tmp.push_move(mv)
                if boards_equal(tmp, dst):
                    return True
            return False

        # Decode current board (already attempted above)
        try:
            board_current = tokenizer.decode(current_tokens.detach().cpu().tolist())
        except Exception:
            board_current = None

        legal_transition = False
        if reverse_backend == "yaneuraou":
            from posdiffusion.returnable.yaneuraou_bindings import returnable_mask

            pred_np = argmax_tokens.detach().cpu().numpy()[None, :]
            src_np = current_tokens.detach().cpu().numpy()[None, :]
            try:
                legal_transition = bool(
                    returnable_mask(
                        pred_np,
                        src_np,
                        use_light_checks=returnable_use_light_checks,
                    )[0]
                )
            except Exception:
                legal_transition = False
        elif board_current is not None:
            try:
                legal_transition = _exists_forward_legal_move(pred_prev_board, board_current)
            except Exception:
                legal_transition = False

        validator_legal = validator.is_valid_position(pred_prev_board) and legal_transition and (change_count in (2, 3))
        step_record = {
            "game_id": game_id,
            "start_move": start_move,
            "step": step_idx,
            "validator_legal": validator_legal,
            "token_change_count": change_count,
            "total_pieces_ok": count_total_pieces(pred_prev_board) == 40,
            "has_double_pawn": has_double_pawn(pred_prev_board),
            "current_sfen": current_sfen,
            "pred_prev_sfen": safe_sfen(pred_prev_board),
        }
        step_logs.append(step_record)
        if not validator_legal:
            any_illegal = True
            illegal_count += 1
            if first_illegal_step is None:
                first_illegal_step = step_idx
            # Stop rewinding immediately on first illegal
            stop_reason = "illegal_detected"
            break

        # Step accepted (only when legal)
        steps_success += 1

        # If reached the initial position, stop
        if boards_equal(pred_prev_board, initial_board):
            current_tokens = torch.tensor(positions[0], device=device, dtype=torch.long)
            stop_reason = "initial_reached"
            break

        # Continue rewinding
        # Encode predicted prev board back to tokens using tokenizer
        current_tokens = torch.tensor(
            np.array(tokenizer.encode(pred_prev_board), dtype=np.int64),
            device=device,
            dtype=torch.long,
        )

    # Final check whether current equals initial
    try:
        final_board = tokenizer.decode(current_tokens.detach().cpu().tolist())
        reached_initial = boards_equal(final_board, initial_board)
        final_sfen = safe_sfen(final_board)
    except Exception:
        reached_initial = False
        final_sfen = None

    return (
        RewindRow(
            game_id=game_id,
            start_move=start_move,
            steps_attempted=steps_attempted,
            steps_success=steps_success,
            reached_initial=bool(reached_initial),
            any_illegal_pred=any_illegal,
            illegal_count=illegal_count,
            first_illegal_step=first_illegal_step,
            stop_reason=stop_reason,
            final_sfen=final_sfen,
        ),
        step_logs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewind to initial via repeated argmax distance=1")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--reverse_backend", type=str, choices=["rsshogi", "yaneuraou"], default="rsshogi")
    parser.add_argument("--no_returnable_light_checks", action="store_true", default=False)
    parser.add_argument("--max_games", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--moves",
        type=str,
        default="10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated start moves to attempt (N).",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_step_logs", action="store_true", help="Do not write per-step JSONL logs")
    args = parser.parse_args()

    device = args.device
    tokenizer = ShogiTokenizer()
    model = load_model(args.checkpoint, device)
    validator = PositionValidator(tokenizer)

    # Parse target moves
    try:
        target_moves = [int(x) for x in args.moves.split(",") if int(x) >= 1]
    except Exception as e:
        raise ValueError(f"Invalid --moves: {args.moves}") from e

    test_pattern = load_test_pattern_from_config(args.config)
    dataset = wds.WebDataset(test_pattern, shardshuffle=False).decode()

    returnable_use_light_checks = not args.no_returnable_light_checks
    rows: list[RewindRow] = []
    steps_all: list[dict[str, Any]] = []
    game_count = 0
    game_iter = maybe_tqdm(dataset, desc="Evaluating games (rewind)", unit="game")
    for sample in game_iter:
        game_id = sample.get("game_id.txt", sample.get("__key__", "unknown"))
        positions = sample["positions.npy"]

        # Try specified moves if within bounds
        for mv in target_moves:
            if mv < len(positions):
                r, step_logs = rewind_from(
                    model=model,
                    tokenizer=tokenizer,
                    validator=validator,
                    device=device,
                    game_id=str(game_id),
                    positions=positions,
                    start_move=mv,
                    reverse_backend=args.reverse_backend,
                    returnable_use_light_checks=returnable_use_light_checks,
                )
                rows.append(r)
                steps_all.extend(step_logs)

        game_count += 1
        if args.max_games and game_count >= args.max_games:
            break

    # Aggregate
    summary: dict[str, Any] = {}
    if rows:
        summary = {
            "num_games": game_count,
            "num_records": len(rows),
            "reached_rate": float(sum(1 for r in rows if r.reached_initial) / len(rows)),
            "illegal_any_rate": float(sum(1 for r in rows if r.any_illegal_pred) / len(rows)),
            "avg_steps_success": float(np.mean([r.steps_success for r in rows])),
        }

    # Output paths
    checkpoint_path = Path(args.checkpoint)
    experiment_dir = checkpoint_path.parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else (experiment_dir / "evaluation_results")
    out_dir.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = out_dir / f"argmax_rewind_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game_id",
                "start_move",
                "steps_attempted",
                "steps_success",
                "reached_initial",
                "any_illegal_pred",
                "illegal_count",
                "first_illegal_step",
                "stop_reason",
                "final_sfen",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    # JSONL per-step logs
    if not args.no_step_logs:
        steps_path = out_dir / f"argmax_rewind_steps_{timestamp}.jsonl"
        with steps_path.open("w", encoding="utf-8") as f:
            for rec in steps_all:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # JSON summary
    json_path = out_dir / f"argmax_rewind_summary_{timestamp}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed games: {game_count}")
    print(f"Records: {len(rows)}")
    print(f"CSV saved to: {csv_path}")
    if not args.no_step_logs:
        print(f"Step logs saved to: {steps_path}")
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
