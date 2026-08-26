"""dlshogi-compatible input feature extraction built on rsshogi.

This module reproduces the standard dlshogi non-nyugyoku feature layout with rsshogi.
The square indexing, piece-type ordering, and hand-piece ordering follow the Apery
layout used by dlshogi-format policy-value models.

Feature layout (per side-to-move perspective):

- ``features1`` shape ``(FEATURES1_NUM=62, 9, 9)``
  - For each color ``c2`` in {own=0, opponent=1}:
    - channels ``[0, 14)``   : piece placement for the 14 piece types (Pawn..Dragon)
    - channels ``[14, 28)``  : merged piece attacks (利き) for the 14 piece types
    - channels ``[28, 31)``  : attacker-count buckets (1, 2, 3+ attackers)
  - own color occupies channels ``[0, 31)``, opponent ``[31, 62)``
- ``features2`` shape ``(FEATURES2_NUM=57, 9, 9)``
  - channels ``[0, 28)``  : own hand pieces (unary encoded)
  - channels ``[28, 56)`` : opponent hand pieces (unary encoded)
  - channel  ``56``       : 1 everywhere if the side to move is in check
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray
from rsshogi.core import Board
from rsshogi.types import Color, PieceType, Square

# dlshogi feature dimensions (non-nyugyoku).
_PIECE_TYPE_NUM: Final[int] = 14  # Pawn..Dragon
_MAX_ATTACK_NUM: Final[int] = 3
_FEATURES1_PER_COLOR: Final[int] = _PIECE_TYPE_NUM + _PIECE_TYPE_NUM + _MAX_ATTACK_NUM  # 31
FEATURES1_NUM: Final[int] = 2 * _FEATURES1_PER_COLOR  # 62

# Hand piece order and per-piece maxima (Apery order).
_HAND_PIECES: Final[tuple[tuple[PieceType, int], ...]] = (
    (PieceType.PAWN, 8),
    (PieceType.LANCE, 4),
    (PieceType.KNIGHT, 4),
    (PieceType.SILVER, 4),
    (PieceType.GOLD, 4),
    (PieceType.BISHOP, 2),
    (PieceType.ROOK, 2),
)
_MAX_PIECES_IN_HAND_SUM: Final[int] = sum(cap for _, cap in _HAND_PIECES)  # 28
_FEATURES2_HAND_NUM: Final[int] = 2 * _MAX_PIECES_IN_HAND_SUM  # 56
FEATURES2_NUM: Final[int] = _FEATURES2_HAND_NUM + 1  # 57

# Cumulative channel offset of each hand piece within one color's 28 channels.
_HAND_OFFSETS: Final[tuple[int, ...]] = tuple(sum(cap for _, cap in _HAND_PIECES[:i]) for i in range(len(_HAND_PIECES)))

# Piece types used for board placement / attack features (Pawn..Dragon).
_PLACEMENT_PIECE_TYPES: Final[tuple[PieceType, ...]] = tuple(PieceType(v) for v in range(1, _PIECE_TYPE_NUM + 1))

# Precomputed Square objects for indices 0..80.
_SQUARES: Final[tuple[Square, ...]] = tuple(Square.from_index(i) for i in range(81))

_COLORS: Final[tuple[Color, Color]] = (Color.BLACK, Color.WHITE)


def make_input_features(board: Board, features1: NDArray[np.float32], features2: NDArray[np.float32]) -> None:
    """Fill dlshogi input features for ``board`` into ``features1`` / ``features2``.

    Args:
        board: rsshogi Board to extract features from.
        features1: Pre-allocated array of shape (FEATURES1_NUM, 9, 9); overwritten.
        features2: Pre-allocated array of shape (FEATURES2_NUM, 9, 9); overwritten.
    """
    features1.fill(0)
    features2.fill(0)

    # Views that index features as (channel, square) over the flattened 9x9 plane.
    f1 = features1.reshape(2, _FEATURES1_PER_COLOR, 81)
    f2 = features2.reshape(FEATURES2_NUM, 81)

    turn = board.turn
    black_to_move = turn.is_black()

    for c in _COLORS:
        # Side-to-move perspective: own color -> channel block 0, opponent -> block 1.
        c2 = 0 if c == turn else 1

        # Piece placement and merged attacks per piece type.
        for pt_idx, pt in enumerate(_PLACEMENT_PIECE_TYPES):
            placement_bb = board.pieces_for(pt, c)
            attack_bb = board.attacks_by(c, pt)
            for sq_idx, sq in enumerate(_SQUARES):
                sq2 = sq_idx if black_to_move else 80 - sq_idx
                if placement_bb.test(sq):
                    f1[c2, pt_idx, sq2] = 1.0
                if attack_bb.test(sq):
                    f1[c2, _PIECE_TYPE_NUM + pt_idx, sq2] = 1.0

        # Attacker-count buckets per square (unary, capped at MAX_ATTACK_NUM).
        for sq_idx, sq in enumerate(_SQUARES):
            sq2 = sq_idx if black_to_move else 80 - sq_idx
            num = min(_MAX_ATTACK_NUM, board.attackers_to_color_current(c, sq).count())
            for k in range(num):
                f1[c2, 2 * _PIECE_TYPE_NUM + k, sq2] = 1.0

        # Hand pieces (unary encoded, flattened across the whole board plane).
        hand = board.hand(c)
        hand_base = c2 * _MAX_PIECES_IN_HAND_SUM
        for (pt, cap), offset in zip(_HAND_PIECES, _HAND_OFFSETS, strict=True):
            count = min(hand.count(pt), cap)
            for k in range(count):
                f2[hand_base + offset + k, :] = 1.0

    # Check indicator.
    if board.is_in_check():
        f2[_FEATURES2_HAND_NUM, :] = 1.0
