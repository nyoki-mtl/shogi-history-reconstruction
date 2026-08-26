from pathlib import Path

import pytest
from rsshogi.core import Board

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.reverse.candidates import generate_predecessor_candidates


def _core_sfen(board: Board) -> str:
    return " ".join(board.to_sfen().split()[:3])


def test_generate_predecessor_candidates_contains_initial_position_after_one_move() -> None:
    tokenizer = ShogiTokenizer(flip_to_black=False)
    predecessor = Board()
    successor = Board(predecessor.to_sfen())
    successor.push_move(next(iter(successor.legal_moves())))

    predecessor_tokens = tuple(tokenizer.encode(predecessor))
    candidates = generate_predecessor_candidates(tokenizer.encode(successor), tokenizer=tokenizer)

    assert predecessor_tokens in {candidate.tokens for candidate in candidates}


def test_generate_predecessor_candidates_are_forward_legal() -> None:
    tokenizer = ShogiTokenizer(flip_to_black=False)
    successor = Board()
    successor.push_move(next(iter(successor.legal_moves())))
    successor_core = _core_sfen(successor)

    candidates = generate_predecessor_candidates(tokenizer.encode(successor), tokenizer=tokenizer)

    assert candidates
    for candidate in candidates:
        predecessor = tokenizer.decode(list(candidate.tokens))
        reached = False
        for move in predecessor.legal_moves():
            test_board = Board(predecessor.to_sfen())
            test_board.push_move(move)
            if _core_sfen(test_board) == successor_core:
                reached = True
                break
        assert reached


def test_yaneuraou_candidate_backend_matches_rsshogi_tokens() -> None:
    lib_path = Path("external/YaneuraOu/build/libyaneuraou_returnable.so")
    if not lib_path.exists():
        pytest.skip("YaneuraOu returnable library is not built")

    tokenizer = ShogiTokenizer(flip_to_black=False)
    board = Board()
    board.push_move(next(iter(board.legal_moves())))
    board.push_move(next(iter(board.legal_moves())))

    successor_tokens = tokenizer.encode(board)
    rsshogi_candidates = generate_predecessor_candidates(successor_tokens, tokenizer=tokenizer, backend="rsshogi")
    yaneuraou_candidates = generate_predecessor_candidates(successor_tokens, tokenizer=tokenizer, backend="yaneuraou")

    assert {candidate.tokens for candidate in yaneuraou_candidates} == {
        candidate.tokens for candidate in rsshogi_candidates
    }
    assert {candidate.move_usi for candidate in yaneuraou_candidates} == {
        candidate.move_usi for candidate in rsshogi_candidates
    }
