"""Fast returnable checks using rsshogi move generation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rsshogi.core import Board
from rsshogi.types import Color, PieceType, Square

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.board.validator import PositionValidator

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


def _piece_type(board: Board, square: int) -> int:
    return _piece(board, square) % 16


def _hand(board: Board, color: Color) -> list[int]:
    return [board.hand(color).count(piece_type) for piece_type in HAND_PIECE_TYPES]


def _has_double_pawn(board: Board, color: Color) -> bool:
    seen_files: set[int] = set()
    for sq in range(81):
        piece = _piece(board, sq)
        if piece == 0:
            continue
        is_white = piece >= 16
        if (color == Color.WHITE) != is_white:
            continue
        if _piece_type(board, sq) != 1:
            continue
        file_idx = sq // 9
        if file_idx in seen_files:
            return True
        seen_files.add(file_idx)
    return False


def _has_dead_piece(board: Board, color: Color) -> bool:
    for sq in range(81):
        piece = _piece(board, sq)
        if piece == 0:
            continue
        is_white = piece >= 16
        if (color == Color.WHITE) != is_white:
            continue
        piece_type = _piece_type(board, sq)
        rank_idx = sq % 9
        if color == Color.BLACK:
            if piece_type in (1, 2, 3) and rank_idx == 0:
                return True
            if piece_type == 3 and rank_idx == 1:
                return True
        else:
            if piece_type == 3 and rank_idx == 7:
                return True
            if piece_type in (1, 2, 3) and rank_idx == 8:
                return True
    return False


def _diff_count(prev: Board, curr: Board) -> int:
    board_diff = 0
    for sq in range(81):
        if _piece(prev, sq) != _piece(curr, sq):
            board_diff += 1
    prev_hands = [_hand(prev, Color.BLACK), _hand(prev, Color.WHITE)]
    curr_hands = [_hand(curr, Color.BLACK), _hand(curr, Color.WHITE)]
    hand_diff = 0
    for side in range(2):
        for idx in range(7):
            hand_diff += abs(prev_hands[side][idx] - curr_hands[side][idx])
    return board_diff + hand_diff


def _sfen_core(board: Board) -> str:
    return " ".join(board.to_sfen().split()[:3])


def _light_checks(prev: Board, curr: Board, validator: PositionValidator) -> bool:
    if not validator.is_valid_position(prev):
        return False
    if not validator.is_valid_position(curr):
        return False
    if _has_double_pawn(prev, Color.BLACK) or _has_double_pawn(prev, Color.WHITE):
        return False
    if _has_double_pawn(curr, Color.BLACK) or _has_double_pawn(curr, Color.WHITE):
        return False
    if _has_dead_piece(prev, Color.BLACK) or _has_dead_piece(prev, Color.WHITE):
        return False
    if _has_dead_piece(curr, Color.BLACK) or _has_dead_piece(curr, Color.WHITE):
        return False
    diff = _diff_count(prev, curr)
    return diff in (2, 3)


def _can_reach_by_legal_move(prev: Board, curr: Board) -> bool:
    target = _sfen_core(curr)
    for mv in prev.legal_moves():
        prev.push_move(mv)
        if _sfen_core(prev) == target:
            prev.pop()
            return True
        prev.pop()
    return False


def returnable_mask(
    predicted_tokens: np.ndarray | Iterable[Iterable[int]],
    source_tokens: np.ndarray | Iterable[Iterable[int]],
    *,
    use_light_checks: bool = True,
) -> np.ndarray:
    tokens_pred = np.asarray(predicted_tokens, dtype=np.int32)
    tokens_src = np.asarray(source_tokens, dtype=np.int32)
    if tokens_pred.shape != tokens_src.shape:
        raise ValueError("predicted_tokens and source_tokens must have the same shape")
    if tokens_pred.ndim != 2 or tokens_pred.shape[1] != 96:
        raise ValueError("tokens must have shape (batch, 96)")

    tokenizer = ShogiTokenizer(flip_to_black=False)
    validator = PositionValidator(tokenizer)

    batch = tokens_pred.shape[0]
    mask = np.zeros(batch, dtype=bool)
    for idx in range(batch):
        prev_board = tokenizer.decode(tokens_pred[idx].tolist())
        curr_board = tokenizer.decode(tokens_src[idx].tolist())
        if use_light_checks and not _light_checks(prev_board, curr_board, validator):
            continue
        if _can_reach_by_legal_move(prev_board, curr_board):
            mask[idx] = True
    return mask
