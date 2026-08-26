"""エンコーディング/デコーディング用のユーティリティ関数。"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from posdiffusion.board.constants import HAND_PIECE_OFFSET, MAX_PAWN_IN_HAND, PIECE_FLIP_DICT


def create_reverse_piece_function() -> Callable[[Any], Any]:
    """駒の先後を反転する関数を作成。

    Returns:
        numpy のベクトル化された変換関数
    """
    # numpy.vectorize returns a numpy.vectorize object that is callable. We cast
    # here to a generic Callable[[Any], Any] expected by callers.
    return cast(Callable[[Any], Any], np.vectorize(PIECE_FLIP_DICT.get))


def num_to_hand_piece_id(num: int, max_pawn_num: int = MAX_PAWN_IN_HAND, offset: int = HAND_PIECE_OFFSET) -> int:
    """持駒の数を持駒IDに変換。

    Args:
        num: 持駒の数
        max_pawn_num: 歩の最大持駒数
        offset: 持駒IDのオフセット

    Returns:
        持駒ID (31以降)
    """
    if num <= max_pawn_num:
        return offset + num
    else:
        # 最大数を超える場合は最大値でクリップ
        return offset + max_pawn_num


def hand_piece_id_to_num(pid: int, max_pawn_num: int = MAX_PAWN_IN_HAND, offset: int = HAND_PIECE_OFFSET) -> int:
    """持駒IDを持駒の数に変換。

    Args:
        pid: 持駒ID
        max_pawn_num: 歩の最大持駒数
        offset: 持駒IDのオフセット

    Returns:
        持駒の数
    """
    num = pid - offset
    if num < 0:
        return 0
    elif num > max_pawn_num:
        return max_pawn_num
    else:
        return num


def create_id_to_num_dict(max_pawn_num: int = MAX_PAWN_IN_HAND, offset: int = HAND_PIECE_OFFSET) -> dict[int, int]:
    """持駒IDから数への変換辞書を作成。"""
    return {offset + idx: idx for idx in range(max_pawn_num + 1)}
