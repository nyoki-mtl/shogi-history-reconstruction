"""Legal predecessor candidate generation.

The generator works from a successor position only.  It enumerates plausible
reverse edits for the last move, decodes them as predecessor positions, and
keeps only candidates that rsshogi confirms can legally move to the successor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from rsshogi.core import Board
from rsshogi.types import Square

from posdiffusion.board.constants import (
    BBISHOP,
    BGOLD,
    BKNIGHT,
    BLACK,
    BLANCE,
    BPAWN,
    BPROM_BISHOP,
    BPROM_KNIGHT,
    BPROM_LANCE,
    BPROM_PAWN,
    BPROM_ROOK,
    BPROM_SILVER,
    BROOK,
    BSILVER,
    WBISHOP,
    WGOLD,
    WHITE,
    WKNIGHT,
    WLANCE,
    WPAWN,
    WPROM_BISHOP,
    WPROM_KNIGHT,
    WPROM_LANCE,
    WPROM_PAWN,
    WPROM_ROOK,
    WPROM_SILVER,
    WROOK,
    WSILVER,
)
from posdiffusion.board.constants.tokenizer import BOARD_SIZE, TOTAL_TOKENS, TURN_BLACK, TURN_WHITE
from posdiffusion.board.encoding_utils import hand_piece_id_to_num, num_to_hand_piece_id
from posdiffusion.board.tokenizer import ShogiTokenizer

if TYPE_CHECKING:
    from posdiffusion.board.validator import PositionValidator

CandidateKind = Literal["move", "drop"]
CandidateBackend = Literal["rsshogi", "yaneuraou", "yaneuraou_hybrid"]

BLACK_UNPROMOTED = (BPAWN, BLANCE, BKNIGHT, BSILVER, BGOLD, BBISHOP, BROOK)
WHITE_UNPROMOTED = (WPAWN, WLANCE, WKNIGHT, WSILVER, WGOLD, WBISHOP, WROOK)
BLACK_CAPTURABLE = (
    BPAWN,
    BLANCE,
    BKNIGHT,
    BSILVER,
    BBISHOP,
    BROOK,
    BGOLD,
    BPROM_PAWN,
    BPROM_LANCE,
    BPROM_KNIGHT,
    BPROM_SILVER,
    BPROM_BISHOP,
    BPROM_ROOK,
)
WHITE_CAPTURABLE = (
    WPAWN,
    WLANCE,
    WKNIGHT,
    WSILVER,
    WBISHOP,
    WROOK,
    WGOLD,
    WPROM_PAWN,
    WPROM_LANCE,
    WPROM_KNIGHT,
    WPROM_SILVER,
    WPROM_BISHOP,
    WPROM_ROOK,
)

DEMOTE = {
    BPROM_PAWN: BPAWN,
    BPROM_LANCE: BLANCE,
    BPROM_KNIGHT: BKNIGHT,
    BPROM_SILVER: BSILVER,
    BPROM_BISHOP: BBISHOP,
    BPROM_ROOK: BROOK,
    WPROM_PAWN: WPAWN,
    WPROM_LANCE: WLANCE,
    WPROM_KNIGHT: WKNIGHT,
    WPROM_SILVER: WSILVER,
    WPROM_BISHOP: WBISHOP,
    WPROM_ROOK: WROOK,
}

UNPROMOTE = {
    BPROM_PAWN: BPAWN,
    BPROM_LANCE: BLANCE,
    BPROM_KNIGHT: BKNIGHT,
    BPROM_SILVER: BSILVER,
    BPROM_BISHOP: BBISHOP,
    BPROM_ROOK: BROOK,
    WPROM_PAWN: WPAWN,
    WPROM_LANCE: WLANCE,
    WPROM_KNIGHT: WKNIGHT,
    WPROM_SILVER: WSILVER,
    WPROM_BISHOP: WBISHOP,
    WPROM_ROOK: WROOK,
}

HAND_TYPE_BY_BASE = {
    BPAWN: 0,
    WPAWN: 0,
    BLANCE: 1,
    WLANCE: 1,
    BKNIGHT: 2,
    WKNIGHT: 2,
    BSILVER: 3,
    WSILVER: 3,
    BGOLD: 4,
    WGOLD: 4,
    BBISHOP: 5,
    WBISHOP: 5,
    BROOK: 6,
    WROOK: 6,
}

DROP_PIECES_BY_COLOR = {
    BLACK: BLACK_UNPROMOTED,
    WHITE: WHITE_UNPROMOTED,
}


@dataclass(frozen=True)
class PredecessorCandidate:
    """A legal predecessor that reaches the successor by one legal move."""

    tokens: tuple[int, ...]
    move_usi: str
    kind: CandidateKind
    from_square: int | None
    to_square: int
    moved_piece: int
    captured_piece: int | None
    promoted: bool


@dataclass(frozen=True)
class _CandidateDraft:
    tokens: tuple[int, ...]
    kind: CandidateKind
    from_square: int | None
    to_square: int
    moved_piece: int
    captured_piece: int | None
    promoted: bool


def _piece(board: Board, square: int) -> int:
    return int(board.piece_on(Square.from_index(square)))


def _sfen_core(board: Board) -> str:
    return " ".join(board.to_sfen().split()[:3])


def _boards_equal(a: Board, b: Board) -> bool:
    return _sfen_core(a) == _sfen_core(b)


def _color_of(piece: int) -> int | None:
    if 1 <= piece <= 14:
        return BLACK
    if 17 <= piece <= 30:
        return WHITE
    return None


def _opposite_color(color: int) -> int:
    return WHITE if color == BLACK else BLACK


def _turn_token_for_color(color: int) -> int:
    return TURN_BLACK if color == BLACK else TURN_WHITE


def _mover_color_from_successor(tokens: list[int]) -> int:
    successor_turn = BLACK if tokens[95] == TURN_BLACK else WHITE
    return _opposite_color(successor_turn)


def _hand_index(color: int, hand_type: int) -> int:
    return 81 + hand_type if color == BLACK else 88 + hand_type


def _hand_count(tokens: list[int], color: int, hand_type: int) -> int:
    return hand_piece_id_to_num(tokens[_hand_index(color, hand_type)])


def _set_hand_count(tokens: list[int], color: int, hand_type: int, count: int) -> None:
    tokens[_hand_index(color, hand_type)] = num_to_hand_piece_id(count)


def _decrement_hand(tokens: list[int], color: int, base_piece: int) -> bool:
    hand_type = HAND_TYPE_BY_BASE[base_piece]
    count = _hand_count(tokens, color, hand_type)
    if count <= 0:
        return False
    _set_hand_count(tokens, color, hand_type, count - 1)
    return True


def _increment_hand(tokens: list[int], color: int, base_piece: int) -> bool:
    hand_type = HAND_TYPE_BY_BASE[base_piece]
    count = _hand_count(tokens, color, hand_type)
    _set_hand_count(tokens, color, hand_type, count + 1)
    return True


def _captured_base(piece: int) -> int:
    return DEMOTE.get(piece, piece)


def _candidate_moved_pieces(successor_piece: int) -> tuple[int, ...]:
    unpromoted = UNPROMOTE.get(successor_piece)
    if unpromoted is None:
        return (successor_piece,)
    return (successor_piece, unpromoted)


def _capturable_pieces(opponent_color: int) -> tuple[int, ...]:
    return BLACK_CAPTURABLE if opponent_color == BLACK else WHITE_CAPTURABLE


def _find_forward_move(prev: Board, successor: Board) -> str | None:
    target = _sfen_core(successor)
    for move in prev.legal_moves():
        prev.push_move(move)
        if _sfen_core(prev) == target:
            move_usi = move.to_usi()
            prev.pop()
            return move_usi
        prev.pop()
    return None


def _passes_position_checks(tokens: list[int], tokenizer: ShogiTokenizer, validator: PositionValidator) -> bool:
    predecessor = tokenizer.decode(tokens)
    if not validator.is_valid_position(predecessor):
        return False
    if not predecessor.is_valid():
        return False
    opponent_turn_tokens = tokens.copy()
    opponent_turn_tokens[95] = TURN_WHITE if tokens[95] == TURN_BLACK else TURN_BLACK
    opponent_view = tokenizer.decode(opponent_turn_tokens)
    return not opponent_view.is_in_check()


def _try_add_draft(
    drafts: list[_CandidateDraft],
    seen: set[tuple[int, ...]],
    tokenizer: ShogiTokenizer,
    validator: PositionValidator | None,
    tokens: list[int],
    *,
    kind: CandidateKind,
    from_square: int | None,
    to_square: int,
    moved_piece: int,
    captured_piece: int | None,
    promoted: bool,
    validate_position: bool,
) -> None:
    key = tuple(tokens)
    if key in seen:
        return

    if validate_position:
        if validator is None:
            raise RuntimeError("PositionValidator is required when validate_position is true")
        if not _passes_position_checks(tokens, tokenizer, validator):
            return

    seen.add(key)
    drafts.append(
        _CandidateDraft(
            tokens=key,
            kind=kind,
            from_square=from_square,
            to_square=to_square,
            moved_piece=moved_piece,
            captured_piece=captured_piece,
            promoted=promoted,
        )
    )


def _verify_drafts_rsshogi(
    drafts: list[_CandidateDraft],
    tokenizer: ShogiTokenizer,
    successor: Board,
) -> list[PredecessorCandidate]:
    candidates: list[PredecessorCandidate] = []
    for draft in drafts:
        predecessor = tokenizer.decode(list(draft.tokens))
        move_usi = _find_forward_move(predecessor, successor)
        if move_usi is None:
            continue
        candidates.append(
            PredecessorCandidate(
                tokens=draft.tokens,
                move_usi=move_usi,
                kind=draft.kind,
                from_square=draft.from_square,
                to_square=draft.to_square,
                moved_piece=draft.moved_piece,
                captured_piece=draft.captured_piece,
                promoted=draft.promoted,
            )
        )
    return candidates


def _verify_drafts_yaneuraou(
    drafts: list[_CandidateDraft],
    successor_tokens: list[int],
) -> list[PredecessorCandidate]:
    from posdiffusion.returnable.yaneuraou_bindings import returnable_moves

    if not drafts:
        return []

    predecessor_array = np.asarray([draft.tokens for draft in drafts], dtype=np.int32)
    successor_array = np.repeat(np.asarray([successor_tokens], dtype=np.int32), repeats=len(drafts), axis=0)
    mask, moves = returnable_moves(predecessor_array, successor_array, use_light_checks=True)

    candidates: list[PredecessorCandidate] = []
    for draft, is_returnable, move_usi in zip(drafts, mask, moves, strict=True):
        if not is_returnable:
            continue
        candidates.append(
            PredecessorCandidate(
                tokens=draft.tokens,
                move_usi=move_usi,
                kind=draft.kind,
                from_square=draft.from_square,
                to_square=draft.to_square,
                moved_piece=draft.moved_piece,
                captured_piece=draft.captured_piece,
                promoted=draft.promoted,
            )
        )
    return candidates


def _generate_predecessor_candidates_yaneuraou(
    successor_tokens: list[int],
) -> list[PredecessorCandidate]:
    from posdiffusion.returnable.yaneuraou_bindings import predecessor_candidates

    token_array, moves, metadata = predecessor_candidates(successor_tokens)
    candidates: list[PredecessorCandidate] = []
    for idx, move_usi in enumerate(moves):
        captured_piece = int(metadata["captured_piece"][idx])
        candidates.append(
            PredecessorCandidate(
                tokens=tuple(int(token) for token in token_array[idx]),
                move_usi=move_usi,
                kind="drop" if int(metadata["kind"][idx]) == 1 else "move",
                from_square=None if int(metadata["from_square"][idx]) < 0 else int(metadata["from_square"][idx]),
                to_square=int(metadata["to_square"][idx]),
                moved_piece=int(metadata["moved_piece"][idx]),
                captured_piece=None if captured_piece < 0 else captured_piece,
                promoted=bool(metadata["promoted"][idx]),
            )
        )
    return candidates


def generate_predecessor_candidates(
    successor_tokens: list[int] | NDArray[np.integer],
    *,
    tokenizer: ShogiTokenizer | None = None,
    backend: CandidateBackend = "rsshogi",
) -> list[PredecessorCandidate]:
    """Enumerate legal predecessor candidates for one successor position.

    The returned candidates are deduplicated by their 96-token predecessor
    representation and verified by the selected legal move backend.
    """

    tokens = [int(token) for token in successor_tokens]
    if len(tokens) != TOTAL_TOKENS:
        raise ValueError(f"Expected {TOTAL_TOKENS} tokens, got {len(tokens)}")
    if backend not in {"rsshogi", "yaneuraou", "yaneuraou_hybrid"}:
        raise ValueError(f"Unknown candidate backend: {backend}")
    if backend == "yaneuraou":
        return _generate_predecessor_candidates_yaneuraou(tokens)

    tokenizer = tokenizer or ShogiTokenizer(flip_to_black=False)
    validate_drafts = backend == "rsshogi"
    if validate_drafts:
        from posdiffusion.board.validator import PositionValidator

        validator = PositionValidator(tokenizer)
    else:
        validator = None
    successor = tokenizer.decode(tokens) if backend == "rsshogi" else None
    mover_color = _mover_color_from_successor(tokens)
    opponent_color = _opposite_color(mover_color)

    drafts: list[_CandidateDraft] = []
    seen: set[tuple[int, ...]] = set()
    empty_squares = [idx for idx, piece in enumerate(tokens[:BOARD_SIZE]) if piece == 0 or piece % 16 == 15]

    for dst, successor_piece in enumerate(tokens[:BOARD_SIZE]):
        if _color_of(successor_piece) != mover_color:
            continue

        for src in empty_squares:
            if src == dst:
                continue
            for moved_piece in _candidate_moved_pieces(successor_piece):
                promoted = moved_piece != successor_piece

                no_capture = tokens.copy()
                no_capture[95] = _turn_token_for_color(mover_color)
                no_capture[src] = moved_piece
                no_capture[dst] = 0
                _try_add_draft(
                    drafts,
                    seen,
                    tokenizer,
                    validator,
                    no_capture,
                    kind="move",
                    from_square=src,
                    to_square=dst,
                    moved_piece=moved_piece,
                    captured_piece=None,
                    promoted=promoted,
                    validate_position=validate_drafts,
                )

                for captured_piece in _capturable_pieces(opponent_color):
                    capture = tokens.copy()
                    capture[95] = _turn_token_for_color(mover_color)
                    capture[src] = moved_piece
                    capture[dst] = captured_piece
                    captured_base = _captured_base(captured_piece)
                    if not _decrement_hand(capture, mover_color, captured_base):
                        continue
                    _try_add_draft(
                        drafts,
                        seen,
                        tokenizer,
                        validator,
                        capture,
                        kind="move",
                        from_square=src,
                        to_square=dst,
                        moved_piece=moved_piece,
                        captured_piece=captured_piece,
                        promoted=promoted,
                        validate_position=validate_drafts,
                    )

        if successor_piece in DROP_PIECES_BY_COLOR[mover_color]:
            drop = tokens.copy()
            drop[95] = _turn_token_for_color(mover_color)
            drop[dst] = 0
            if _increment_hand(drop, mover_color, successor_piece):
                _try_add_draft(
                    drafts,
                    seen,
                    tokenizer,
                    validator,
                    drop,
                    kind="drop",
                    from_square=None,
                    to_square=dst,
                    moved_piece=successor_piece,
                    captured_piece=None,
                    promoted=False,
                    validate_position=validate_drafts,
                )

    if backend == "rsshogi":
        if successor is None:
            raise RuntimeError("rsshogi backend requires a decoded successor")
        return _verify_drafts_rsshogi(drafts, tokenizer, successor)
    return _verify_drafts_yaneuraou(drafts, tokens)


def candidate_tokens_array(candidates: list[PredecessorCandidate]) -> NDArray[np.int32]:
    """Return candidate predecessor tokens as an array with shape (N, 96)."""

    return np.asarray([candidate.tokens for candidate in candidates], dtype=np.int32)
