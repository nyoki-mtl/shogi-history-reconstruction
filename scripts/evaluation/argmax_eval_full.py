"""Full evaluation over test dataset using argmax distance=1 predictions.

For each position (N) in test shards, predict previous position (N-1) via argmax
and compute the following metrics:
  - total_pieces_ok: predicted board + hands totals to 40 pieces
  - has_double_pawn: whether any side has double unpromoted pawn on same file
  - validator_legal: PositionValidator-based legality
  - token_change_count: number of token changes excluding turn token (index 95)
  - reverse_move_exists: existence of a legal move from predicted (N-1) to current (N)
  - alphazero_prob_for_reverse: AlphaZero policy probability for that move (if exists)

Outputs:
  - CSV with per-position metrics (optionally subsampled)
  - JSON summary with aggregated statistics

Usage:
  python scripts/evaluation/argmax_eval_full.py \
    --checkpoint path/to/ckpt.ckpt \
    --alphazero_onnx path/to/model.onnx \
    --config configs/base.yaml \
    --device cuda \
    --max_games 100 --max_positions_per_game 0 \
    --csv_limit 0
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
from rsshogi.core import Board
from rsshogi.types import Color, PieceType, Square

from posdiffusion.board.constants import (
    BBISHOP,
    BGOLD,
    BKING,
    BKNIGHT,
    BLANCE,
    BPAWN,
    BROOK,
    BSILVER,
    HPAWN,
    MAX_PAWN_IN_HAND,
    WBISHOP,
    WGOLD,
    WKING,
    WKNIGHT,
    WLANCE,
    WPAWN,
    WROOK,
    WSILVER,
)
from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.board.validator import PositionValidator
from posdiffusion.diffusion.transformer_denoiser import ShogiTransformerDenoiser
from posdiffusion.evaluation.policy.alphazero_onnx import AlphaZeroONNX
from posdiffusion.inference import load_model

try:  # Best-effort progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)


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


def _sfen_core(board: Board) -> str:
    return " ".join(board.to_sfen().split()[:3])


@dataclass
class EvalRow:
    game_id: str
    move_index: int
    current_sfen: str | None
    pred_prev_sfen: str | None
    total_pieces_ok: bool
    has_double_pawn: bool
    validator_legal: bool
    token_change_count: int
    change_is_2_or_3: bool
    reverse_move_exists: bool
    alphazero_prob_for_reverse: float | None


def load_test_pattern_from_config(config_path: str | Path) -> str:
    cfg = OmegaConf.load(config_path)
    pattern = OmegaConf.select(cfg, "data.test_shard_pattern") or OmegaConf.select(cfg, "test_shard_pattern")
    if pattern is None:
        raise ValueError(f"Config {config_path} must contain data.test_shard_pattern")
    return str(pattern)


def count_total_pieces(board: Board) -> int:
    on_board = sum(1 for sq in range(81) if _piece(board, sq) != 0)
    in_hands = 0
    for color in (Color.BLACK, Color.WHITE):
        in_hands += sum(_hand(board, color))
    return on_board + in_hands


def has_double_pawn(board: Board) -> bool:
    # file-major: index = file*9 + rank
    for color in (Color.BLACK, Color.WHITE):
        for file_idx in range(9):
            count = 0
            for rank_idx in range(9):
                sq = file_idx * 9 + rank_idx
                piece = _piece(board, sq)
                if (color == Color.BLACK and piece == BPAWN) or (color == Color.WHITE and piece == WPAWN):
                    count += 1
            if count >= 2:
                return True
    return False


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


def find_reverse_move(pred_prev: Board, current: Board) -> str | None:
    for move in pred_prev.legal_moves():
        test_board = Board(pred_prev.to_sfen())
        test_board.push_move(move)
        if boards_equal(test_board, current):
            return move.to_usi()
    return None


def safe_sfen(board: Board | None) -> str | None:
    if board is None:
        return None
    try:
        return board.to_sfen()
    except Exception as e:  # pragma: no cover
        return f"ERROR:{e}"


def is_valid_position_without_is_ok(validator: PositionValidator, board: Board) -> bool:
    """Validatorのロジックから board.is_ok() を除いた判定。

    - 駒数（成駒を元駒に統合）
    - 王の存在（各1、盤上）
    - MAX_PAWN_IN_HAND 特例下の歩合計
    を満たせば True。
    """
    piece_counts = validator._count_pieces(board)
    merged = validator._merge_promoted_pieces(piece_counts)

    # Kings count
    if merged.get(BKING, 0) != 1 or merged.get(WKING, 0) != 1:
        return False

    # Kings must be on board
    black_king_on_board = any(_piece(board, sq) == BKING for sq in range(81))
    white_king_on_board = any(_piece(board, sq) == WKING for sq in range(81))
    if not black_king_on_board or not white_king_on_board:
        return False

    # Max hand pawns rule
    black_hand_pawns = _hand(board, Color.BLACK)[HPAWN]
    white_hand_pawns = _hand(board, Color.WHITE)[HPAWN]
    has_max_hand_pawns = MAX_PAWN_IN_HAND in {black_hand_pawns, white_hand_pawns}

    piece_totals = {
        "pawn": merged.get(BPAWN, 0) + merged.get(WPAWN, 0),
        "lance": merged.get(BLANCE, 0) + merged.get(WLANCE, 0),
        "knight": merged.get(BKNIGHT, 0) + merged.get(WKNIGHT, 0),
        "silver": merged.get(BSILVER, 0) + merged.get(WSILVER, 0),
        "gold": merged.get(BGOLD, 0) + merged.get(WGOLD, 0),
        "bishop": merged.get(BBISHOP, 0) + merged.get(WBISHOP, 0),
        "rook": merged.get(BROOK, 0) + merged.get(WROOK, 0),
    }

    for piece_type, expected in PositionValidator.TOTAL_PIECE_COUNTS.items():
        actual = piece_totals[piece_type]
        if piece_type == "pawn" and has_max_hand_pawns:
            if actual < expected - 1:
                return False
        elif actual != expected:
            return False

    return True


def evaluate_game(
    model: ShogiTransformerDenoiser,
    tokenizer: ShogiTokenizer,
    validator: PositionValidator,
    alphazero: AlphaZeroONNX | None,
    device: str,
    game_id: str,
    positions: np.ndarray,
    reverse_backend: str,
    returnable_use_light_checks: bool,
    max_positions: int = 0,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    # Determine range of moves to evaluate (from move 1 .. end-1)
    end = len(positions)
    if max_positions and max_positions > 0:
        end = min(end, max_positions + 1)  # +1 because we index prev at i-1
    for move_idx in range(1, end):
        current_tokens = torch.tensor(positions[move_idx], device=device, dtype=torch.long)
        with torch.no_grad():
            logits = model(
                current_tokens.unsqueeze(0),
                torch.ones(1, device=device, dtype=torch.long),
                absolute_move=None,
                game_phase=None,
            )[0]
            argmax_tokens = torch.argmax(logits, dim=-1)

        # Decode predicted prev and compute metrics
        try:
            board_pred_prev = tokenizer.decode(argmax_tokens.detach().cpu().tolist())
            board_current = tokenizer.decode(current_tokens.detach().cpu().tolist())
        except Exception:
            # If decode fails, record obvious negatives
            rows.append(
                EvalRow(
                    game_id=game_id,
                    move_index=move_idx,
                    current_sfen=safe_sfen(None),
                    pred_prev_sfen=None,
                    total_pieces_ok=False,
                    has_double_pawn=False,
                    validator_legal=False,
                    token_change_count=int((current_tokens[:95] != argmax_tokens[:95]).to(torch.int).sum().item()),
                    change_is_2_or_3=False,
                    reverse_move_exists=False,
                    alphazero_prob_for_reverse=None,
                )
            )
            continue

        # SFENs (safe)
        current_sfen = safe_sfen(board_current)
        pred_prev_sfen = safe_sfen(board_pred_prev)

        total_ok = count_total_pieces(board_pred_prev) == 40
        ni_fu = has_double_pawn(board_pred_prev)
        legal_ok = is_valid_position_without_is_ok(validator, board_pred_prev)
        change_count = int((current_tokens[:95] != argmax_tokens[:95]).to(torch.int).sum().item())
        change_ok = change_count in (2, 3)

        rev_move: str | None = None
        rev_exists = False
        if reverse_backend == "yaneuraou":
            from posdiffusion.returnable.yaneuraou_bindings import returnable_moves

            pred_np = argmax_tokens.detach().cpu().numpy()[None, :]
            src_np = current_tokens.detach().cpu().numpy()[None, :]
            mask, moves = returnable_moves(
                pred_np,
                src_np,
                use_light_checks=returnable_use_light_checks,
            )
            rev_exists = bool(mask[0])
            if rev_exists and moves[0]:
                rev_move = moves[0]
        else:
            rev_move = find_reverse_move(board_pred_prev, board_current)
            rev_exists = rev_move is not None

        prob = None
        if alphazero is not None and rev_move is not None:
            move_probs = alphazero.get_legal_move_probs(board_pred_prev)
            prob = float(move_probs.get(rev_move, 0.0))

        rows.append(
            EvalRow(
                game_id=game_id,
                move_index=move_idx,
                current_sfen=current_sfen,
                pred_prev_sfen=pred_prev_sfen,
                total_pieces_ok=total_ok,
                has_double_pawn=ni_fu,
                validator_legal=legal_ok,
                token_change_count=change_count,
                change_is_2_or_3=change_ok,
                reverse_move_exists=rev_exists,
                alphazero_prob_for_reverse=prob,
            )
        )

    return rows


def aggregate(rows: list[EvalRow]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "num_positions": 0,
        }

    def mean_bool(xs: list[bool]) -> float:
        return float(sum(1 for x in xs if x) / len(xs)) if xs else 0.0

    summary: dict[str, Any] = {
        "num_positions": n,
        "rate_total_pieces_ok": mean_bool([r.total_pieces_ok for r in rows]),
        "rate_no_double_pawn": mean_bool([not r.has_double_pawn for r in rows]),
        "rate_validator_legal": mean_bool([r.validator_legal for r in rows]),
        "rate_change_is_2_or_3": mean_bool([r.change_is_2_or_3 for r in rows]),
        "rate_reverse_move_exists": mean_bool([r.reverse_move_exists for r in rows]),
    }
    probs = [r.alphazero_prob_for_reverse for r in rows if r.alphazero_prob_for_reverse is not None]
    if probs:
        arr = np.array(probs, dtype=np.float32)
        summary.update(
            {
                "alphazero_prob_mean": float(arr.mean()),
                "alphazero_prob_median": float(np.median(arr)),
                "alphazero_prob_p90": float(np.quantile(arr, 0.9)),
                "alphazero_prob_count": int(arr.size),
            }
        )
    else:
        summary.update(
            {
                "alphazero_prob_mean": None,
                "alphazero_prob_median": None,
                "alphazero_prob_p90": None,
                "alphazero_prob_count": 0,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full test evaluation with argmax distance=1")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--alphazero_onnx", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--reverse_backend", type=str, choices=["rsshogi", "yaneuraou"], default="rsshogi")
    parser.add_argument("--no_returnable_light_checks", action="store_true", default=False)
    parser.add_argument("--max_games", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max_positions_per_game", type=int, default=0, help="0 means all")
    parser.add_argument("--csv_limit", type=int, default=0, help="0 means all rows in CSV")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    tokenizer = ShogiTokenizer()
    model = load_model(args.checkpoint, device)
    # AlphaZeroはCPUで実行（CUDAプロバイダ未導入環境の警告回避）
    alphazero = AlphaZeroONNX(args.alphazero_onnx, device="cpu") if args.alphazero_onnx else None
    validator = PositionValidator(tokenizer)

    test_pattern = load_test_pattern_from_config(args.config)
    dataset = wds.WebDataset(test_pattern, shardshuffle=False).decode()

    returnable_use_light_checks = not args.no_returnable_light_checks
    all_rows: list[EvalRow] = []
    game_count = 0
    game_iter = tqdm(dataset, desc="Evaluating games (full)", unit="game")
    for sample in game_iter:
        game_id = sample.get("game_id.txt", sample.get("__key__", "unknown"))
        positions = sample["positions.npy"]
        rows = evaluate_game(
            model=model,
            tokenizer=tokenizer,
            validator=validator,
            alphazero=alphazero,
            device=device,
            game_id=str(game_id),
            positions=positions,
            reverse_backend=args.reverse_backend,
            returnable_use_light_checks=returnable_use_light_checks,
            max_positions=args.max_positions_per_game,
        )
        all_rows.extend(rows)
        game_count += 1
        if args.max_games:
            game_iter.set_postfix_str(f"{game_count}/{args.max_games}")
        if args.max_games and game_count >= args.max_games:
            break

    summary = aggregate(all_rows)

    # Output files next to experiment dir by default
    checkpoint_path = Path(args.checkpoint)
    experiment_dir = checkpoint_path.parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else (experiment_dir / "evaluation_results")
    out_dir.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Write CSV (optionally limited)
    csv_path = out_dir / f"argmax_full_{timestamp}.csv"
    limit = args.csv_limit if args.csv_limit and args.csv_limit > 0 else None
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game_id",
                "move_index",
                "current_sfen",
                "pred_prev_sfen",
                "total_pieces_ok",
                "has_double_pawn",
                "validator_legal",
                "token_change_count",
                "change_is_2_or_3",
                "reverse_move_exists",
                "alphazero_prob_for_reverse",
            ],
        )
        writer.writeheader()
        for idx, r in enumerate(all_rows):
            writer.writerow(asdict(r))
            if limit is not None and (idx + 1) >= limit:
                break

    # Write JSON summary
    json_path = out_dir / f"argmax_full_summary_{timestamp}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed games: {game_count}")
    print(f"Positions evaluated: {len(all_rows)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
