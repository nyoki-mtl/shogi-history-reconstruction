"""Search algorithms for legal-projected history reconstruction."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass

from posdiffusion.board.constants.tokenizer import (
    BOARD_TOKEN_END,
    HAND_TOKEN_END,
    HAND_TOKEN_START,
    TOTAL_TOKENS,
    TURN_TOKEN_INDEX,
)
from posdiffusion.board.encoding_utils import hand_piece_id_to_num

Tokens = tuple[int, ...]

# Children are (predecessor tokens, per-step model score), best first.
ExpandFn = Callable[[Tokens], list[tuple[Tokens, float]]]

_PROMOTED_TYPE_MIN = 9
_PROMOTED_TYPE_MAX = 14


def rollback_lower_bound(tokens: Tokens, initial_tokens: Tokens) -> int:
    """Admissible lower bound on reverse steps from ``tokens`` to the initial position.

    Components, each derived from the maximum change one reverse step can make:

    - one move touches at most 2 board squares, so ``ceil(board_diff / 2)``;
    - one move changes the total number of pieces in hand by at most 1 and the
      initial position has empty hands, so the current hand total;
    - one move can unpromote at most one promoted board piece and the initial
      position has none, so the promoted piece count;
    - the step count parity is fixed by the turn token, so round up to parity.
    """

    if len(tokens) != TOTAL_TOKENS or len(initial_tokens) != TOTAL_TOKENS:
        raise ValueError(f"tokens and initial_tokens must have length {TOTAL_TOKENS}")

    board_diff = sum(
        token != initial
        for token, initial in zip(tokens[:BOARD_TOKEN_END], initial_tokens[:BOARD_TOKEN_END], strict=True)
    )
    hand_total = sum(hand_piece_id_to_num(token) for token in tokens[HAND_TOKEN_START:HAND_TOKEN_END])
    promoted = sum(1 for token in tokens[:BOARD_TOKEN_END] if _PROMOTED_TYPE_MIN <= token % 16 <= _PROMOTED_TYPE_MAX)
    bound = max((board_diff + 1) // 2, hand_total, promoted)
    parity = 0 if tokens[TURN_TOKEN_INDEX] == initial_tokens[TURN_TOKEN_INDEX] else 1
    if bound % 2 != parity:
        bound += 1
    return bound


@dataclass
class SearchResult:
    """Outcome of one best-first rollback search."""

    reached_initial: bool
    # States from the start position to the initial position inclusive.
    path: list[Tokens] | None
    steps_to_initial: int | None
    goal_score: float | None
    expansions: int
    generated: int
    stop_reason: str
    start_lower_bound: int
    max_depth_expanded: int
    min_bound_seen: int


@dataclass
class _Node:
    tokens: Tokens
    steps: int
    cum_score: float
    parent: _Node | None


def _reconstruct_path(node: _Node, goal_tokens: Tokens) -> list[Tokens]:
    path = [goal_tokens]
    current: _Node | None = node
    while current is not None:
        path.append(current.tokens)
        current = current.parent
    path.reverse()
    return path


def _is_dominated(entries: list[tuple[int, float]], steps: int, score: float) -> bool:
    return any(prev_steps <= steps and prev_score >= score for prev_steps, prev_score in entries)


def _add_pareto(entries: list[tuple[int, float]], steps: int, score: float) -> None:
    entries[:] = [entry for entry in entries if not (steps <= entry[0] and score >= entry[1])]
    entries.append((steps, score))


def iterative_widening_beam_rollback(
    *,
    start_tokens: Tokens,
    initial_tokens: Tokens,
    expand_fn: ExpandFn,
    max_steps: int,
    expansion_budget: int,
    widths: tuple[int, ...] = (1, 3, 10, 30, 100),
) -> SearchResult:
    """Layer-synchronized beam rollback with increasing width and pruning.

    Best-first search compares cumulative scores across depths, which is poorly
    calibrated when per-step costs are large and variable: deep nodes always
    look worse and the loose midgame lower bound cannot compensate.  Plain beam
    search avoids that by only comparing candidates at the same depth, but it
    cannot backtrack.  This routine restarts the beam with increasing width,
    which acts as coarse-grained backtracking: a wider retry revisits the
    abandoned alternatives of the narrower pass.  Children that cannot reach
    the initial position in the remaining steps are pruned with the admissible
    lower bound, which frees beam slots that plain beam search wastes on doomed
    states.

    ``expansion_budget`` counts unique expanded states across all widths, which
    matches the cost of a run with shared expansion caching.
    """

    if expansion_budget <= 0:
        raise ValueError("expansion_budget must be positive")
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("widths must be positive")

    start_bound = rollback_lower_bound(start_tokens, initial_tokens)
    if start_tokens == initial_tokens:
        return SearchResult(
            reached_initial=True,
            path=[start_tokens],
            steps_to_initial=0,
            goal_score=0.0,
            expansions=0,
            generated=0,
            stop_reason="initial_found",
            start_lower_bound=start_bound,
            max_depth_expanded=0,
            min_bound_seen=0,
        )
    if start_bound > max_steps:
        return SearchResult(
            reached_initial=False,
            path=None,
            steps_to_initial=None,
            goal_score=None,
            expansions=0,
            generated=0,
            stop_reason="unreachable_within_steps",
            start_lower_bound=start_bound,
            max_depth_expanded=0,
            min_bound_seen=start_bound,
        )

    expanded_tokens: set[Tokens] = set()
    expansions = 0
    generated = 0
    max_depth_expanded = 0
    min_bound_seen = start_bound
    budget_exhausted = False

    for width in widths:
        root = _Node(tokens=start_tokens, steps=0, cum_score=0.0, parent=None)
        beam: list[_Node] = [root]
        for step in range(1, max_steps + 1):
            layer: dict[Tokens, _Node] = {}
            for node in beam:
                if expansions >= expansion_budget:
                    budget_exhausted = True
                    break
                if node.tokens not in expanded_tokens:
                    expanded_tokens.add(node.tokens)
                    expansions += 1
                    max_depth_expanded = max(max_depth_expanded, node.steps)
                for child_tokens, delta_score in expand_fn(node.tokens):
                    generated += 1
                    child_score = node.cum_score + delta_score
                    if child_tokens == initial_tokens:
                        return SearchResult(
                            reached_initial=True,
                            path=_reconstruct_path(node, child_tokens),
                            steps_to_initial=step,
                            goal_score=child_score,
                            expansions=expansions,
                            generated=generated,
                            stop_reason="initial_found",
                            start_lower_bound=start_bound,
                            max_depth_expanded=max_depth_expanded,
                            min_bound_seen=0,
                        )
                    bound = rollback_lower_bound(child_tokens, initial_tokens)
                    min_bound_seen = min(min_bound_seen, bound)
                    if step + bound > max_steps:
                        continue
                    existing = layer.get(child_tokens)
                    if existing is None or child_score > existing.cum_score:
                        layer[child_tokens] = _Node(tokens=child_tokens, steps=step, cum_score=child_score, parent=node)
            if budget_exhausted:
                break
            beam = sorted(layer.values(), key=lambda item: item.cum_score, reverse=True)[:width]
            if not beam:
                break
        if budget_exhausted:
            break

    return SearchResult(
        reached_initial=False,
        path=None,
        steps_to_initial=None,
        goal_score=None,
        expansions=expansions,
        generated=generated,
        stop_reason="budget_exhausted" if budget_exhausted else "widths_exhausted",
        start_lower_bound=start_bound,
        max_depth_expanded=max_depth_expanded,
        min_bound_seen=min_bound_seen,
    )


def best_first_rollback(
    *,
    start_tokens: Tokens,
    initial_tokens: Tokens,
    expand_fn: ExpandFn,
    max_steps: int,
    expansion_budget: int,
    step_cost: float = 0.0,
    step_bonus: float = 0.0,
) -> SearchResult:
    """Run an anytime best-first rollback search toward the initial position.

    Nodes are ordered by ``cum_score + step_bonus * steps - step_cost * h(tokens)``
    (higher first), where ``h`` is :func:`rollback_lower_bound`.

    ``step_cost`` rewards heuristic progress toward the initial position, but the
    lower bound is loose in the midgame: from deep start positions, ``h`` is far
    below the true remaining distance, so many necessary steps reduce ``h`` by
    nothing and earn no priority.  ``step_bonus`` rewards depth itself, trading
    breadth (small values, more hedging across alternatives) against depth-first
    commitment (large values approach greedy descent with backtracking).

    Branches with ``steps + h > max_steps`` are pruned, which is safe because
    ``h`` is admissible.  ``expansion_budget`` counts unique candidate-generation
    and model-scoring calls; revisiting an already expanded state is free.
    """

    if expansion_budget <= 0:
        raise ValueError("expansion_budget must be positive")
    if step_cost < 0:
        raise ValueError("step_cost must be non-negative; it scales the heuristic pull")
    if step_bonus < 0:
        raise ValueError("step_bonus must be non-negative; it scales the depth reward")

    start_bound = rollback_lower_bound(start_tokens, initial_tokens)
    if start_tokens == initial_tokens:
        return SearchResult(
            reached_initial=True,
            path=[start_tokens],
            steps_to_initial=0,
            goal_score=0.0,
            expansions=0,
            generated=0,
            stop_reason="initial_found",
            start_lower_bound=start_bound,
            max_depth_expanded=0,
            min_bound_seen=0,
        )
    if start_bound > max_steps:
        return SearchResult(
            reached_initial=False,
            path=None,
            steps_to_initial=None,
            goal_score=None,
            expansions=0,
            generated=0,
            stop_reason="unreachable_within_steps",
            start_lower_bound=start_bound,
            max_depth_expanded=0,
            min_bound_seen=start_bound,
        )

    root = _Node(tokens=start_tokens, steps=0, cum_score=0.0, parent=None)
    counter = 0
    heap: list[tuple[float, int, _Node]] = [(step_cost * start_bound, counter, root)]
    pareto: dict[Tokens, list[tuple[int, float]]] = {start_tokens: [(0, 0.0)]}
    children_cache: dict[Tokens, list[tuple[Tokens, float]]] = {}
    expansions = 0
    generated = 0
    stop_reason = "queue_empty"
    max_depth_expanded = 0
    min_bound_seen = start_bound

    while heap:
        if expansions >= expansion_budget:
            stop_reason = "budget_exhausted"
            break
        _, _, node = heapq.heappop(heap)
        if _is_dominated(pareto.get(node.tokens, []), node.steps, node.cum_score) and (
            (node.steps, node.cum_score) not in pareto.get(node.tokens, [])
        ):
            continue

        children = children_cache.get(node.tokens)
        if children is None:
            children = expand_fn(node.tokens)
            children_cache[node.tokens] = children
            expansions += 1
            max_depth_expanded = max(max_depth_expanded, node.steps)

        for child_tokens, delta_score in children:
            child_steps = node.steps + 1
            child_score = node.cum_score + delta_score
            generated += 1
            if child_tokens == initial_tokens:
                return SearchResult(
                    reached_initial=True,
                    path=_reconstruct_path(node, child_tokens),
                    steps_to_initial=child_steps,
                    goal_score=child_score,
                    expansions=expansions,
                    generated=generated,
                    stop_reason="initial_found",
                    start_lower_bound=start_bound,
                    max_depth_expanded=max_depth_expanded,
                    min_bound_seen=0,
                )
            bound = rollback_lower_bound(child_tokens, initial_tokens)
            min_bound_seen = min(min_bound_seen, bound)
            if child_steps + bound > max_steps:
                continue
            entries = pareto.setdefault(child_tokens, [])
            if _is_dominated(entries, child_steps, child_score):
                continue
            _add_pareto(entries, child_steps, child_score)
            counter += 1
            child = _Node(tokens=child_tokens, steps=child_steps, cum_score=child_score, parent=node)
            priority = child_score + step_bonus * child_steps - step_cost * bound
            heapq.heappush(heap, (-priority, counter, child))

    return SearchResult(
        reached_initial=False,
        path=None,
        steps_to_initial=None,
        goal_score=None,
        expansions=expansions,
        generated=generated,
        stop_reason=stop_reason,
        start_lower_bound=start_bound,
        max_depth_expanded=max_depth_expanded,
        min_bound_seen=min_bound_seen,
    )
