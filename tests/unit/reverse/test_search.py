"""Tests for best-first rollback search and its heuristic lower bound."""

from __future__ import annotations

import pytest
from rsshogi.core import Board

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.reverse.candidates import generate_predecessor_candidates
from posdiffusion.reverse.search import (
    best_first_rollback,
    iterative_widening_beam_rollback,
    rollback_lower_bound,
)

GAME_USI = ["7g7f", "3c3d", "8h2b+", "3a2b", "B*4e", "8c8d", "4e3d", "8d8e"]


@pytest.fixture(scope="module")
def tokenizer() -> ShogiTokenizer:
    return ShogiTokenizer(flip_to_black=False)


@pytest.fixture(scope="module")
def game_tokens(tokenizer: ShogiTokenizer) -> list[tuple[int, ...]]:
    board = Board()
    states = [tuple(tokenizer.encode(board))]
    for usi in GAME_USI:
        board.push_usi(usi)
        states.append(tuple(tokenizer.encode(board)))
    return states


def test_lower_bound_zero_at_initial(game_tokens: list[tuple[int, ...]]) -> None:
    assert rollback_lower_bound(game_tokens[0], game_tokens[0]) == 0


def test_lower_bound_admissible_along_game(game_tokens: list[tuple[int, ...]]) -> None:
    initial = game_tokens[0]
    for ply, tokens in enumerate(game_tokens):
        bound = rollback_lower_bound(tokens, initial)
        assert bound <= ply
        assert bound % 2 == ply % 2


def test_lower_bound_counts_hand_and_promotion(game_tokens: list[tuple[int, ...]]) -> None:
    # After 8h2b+ x captures, both sides hold a bishop and a promoted bishop is on board.
    initial = game_tokens[0]
    assert rollback_lower_bound(game_tokens[3], initial) >= 2


def _expand_with_rsshogi(tokenizer: ShogiTokenizer):
    def expand(tokens: tuple[int, ...]) -> list[tuple[tuple[int, ...], float]]:
        candidates = generate_predecessor_candidates(list(tokens), tokenizer=tokenizer, backend="rsshogi")
        # Deterministic non-model scorer: prefer candidates close to the successor.
        scored = [
            (
                candidate.tokens,
                -float(sum(a != b for a, b in zip(candidate.tokens, tokens, strict=True))),
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    return expand


def test_search_reaches_initial_from_shallow_position(
    tokenizer: ShogiTokenizer, game_tokens: list[tuple[int, ...]]
) -> None:
    start_ply = 4
    result = best_first_rollback(
        start_tokens=game_tokens[start_ply],
        initial_tokens=game_tokens[0],
        expand_fn=_expand_with_rsshogi(tokenizer),
        max_steps=start_ply,
        expansion_budget=200,
        step_cost=1.0,
    )
    assert result.reached_initial
    assert result.stop_reason == "initial_found"
    assert result.path is not None
    assert result.path[0] == game_tokens[start_ply]
    assert result.path[-1] == game_tokens[0]
    assert result.steps_to_initial == start_ply
    assert len(result.path) == start_ply + 1


def test_widening_beam_reaches_initial_from_shallow_position(
    tokenizer: ShogiTokenizer, game_tokens: list[tuple[int, ...]]
) -> None:
    start_ply = 4
    result = iterative_widening_beam_rollback(
        start_tokens=game_tokens[start_ply],
        initial_tokens=game_tokens[0],
        expand_fn=_expand_with_rsshogi(tokenizer),
        max_steps=start_ply,
        expansion_budget=300,
        widths=(1, 3, 10),
    )
    assert result.reached_initial
    assert result.path is not None
    assert result.path[0] == game_tokens[start_ply]
    assert result.path[-1] == game_tokens[0]
    assert result.steps_to_initial is not None
    assert len(result.path) == result.steps_to_initial + 1


def test_search_trivial_at_initial(game_tokens: list[tuple[int, ...]]) -> None:
    result = best_first_rollback(
        start_tokens=game_tokens[0],
        initial_tokens=game_tokens[0],
        expand_fn=lambda tokens: [],
        max_steps=0,
        expansion_budget=1,
    )
    assert result.reached_initial
    assert result.steps_to_initial == 0


def test_search_prunes_unreachable_depth(game_tokens: list[tuple[int, ...]]) -> None:
    result = best_first_rollback(
        start_tokens=game_tokens[4],
        initial_tokens=game_tokens[0],
        expand_fn=lambda tokens: [],
        max_steps=1,
        expansion_budget=10,
    )
    assert not result.reached_initial
    assert result.stop_reason == "unreachable_within_steps"
    assert result.expansions == 0
