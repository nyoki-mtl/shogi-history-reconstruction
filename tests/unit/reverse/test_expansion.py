"""Tests for the shared expansion provider, including the policy blend mode."""

from __future__ import annotations

import pytest
import torch
from rsshogi.core import Board

from posdiffusion.board.constants.tokenizer import TOTAL_TOKENS
from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.reverse.candidates import generate_predecessor_candidates
from posdiffusion.reverse.expansion import BLEND_MODE, POLICY_MODE, ExpansionProvider

VOCAB_SIZE = 40


@pytest.fixture(scope="module")
def tokenizer() -> ShogiTokenizer:
    return ShogiTokenizer(flip_to_black=False)


@pytest.fixture(scope="module")
def successor_tokens(tokenizer: ShogiTokenizer) -> tuple[int, ...]:
    # A position one ply past the initial position has a tractable candidate set.
    board = Board()
    board.push_usi("7g7f")
    board.push_usi("3c3d")
    return tuple(tokenizer.encode(board))


class _ConstantModel:
    """Returns fixed per-position logits so ``changed`` scores are deterministic."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    def __call__(self, source: torch.Tensor, *_args: object, **_kwargs: object) -> torch.Tensor:
        batch = source.shape[0]
        return self._logits.unsqueeze(0).expand(batch, -1, -1)


def _uniform_model() -> _ConstantModel:
    return _ConstantModel(torch.zeros((TOTAL_TOKENS, VOCAB_SIZE), dtype=torch.float32))


def test_changed_mode_matches_reference_provider(tokenizer: ShogiTokenizer, successor_tokens: tuple[int, ...]) -> None:
    """The provider's ``changed`` output equals a direct reference computation."""
    model = _uniform_model()
    provider = ExpansionProvider(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        candidate_backend="rsshogi",
        per_state_topk=5,
        random_seed=0,
    )
    children = provider.expand_fn("changed")(successor_tokens)

    # Reference: generate candidates, score them with the same uniform logits.
    candidates = generate_predecessor_candidates(list(successor_tokens), tokenizer=tokenizer, backend="rsshogi")
    from posdiffusion.reverse.scoring import candidates_to_tensor, score_candidate_tokens

    source = torch.tensor(successor_tokens, dtype=torch.long)
    candidate_tensor = candidates_to_tensor(candidates, device="cpu")
    logits = torch.zeros((TOTAL_TOKENS, VOCAB_SIZE), dtype=torch.float32)
    scores = score_candidate_tokens(logits, source, candidate_tensor, mode="changed").scores
    ranked = sorted(
        zip([float(s) for s in scores.tolist()], [c.tokens for c in candidates], strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    expected = [(candidate, score) for score, candidate in ranked[:5]]

    assert children == expected


def test_blend_demotes_low_policy_candidate(tokenizer: ShogiTokenizer, successor_tokens: tuple[int, ...]) -> None:
    """A candidate with near-zero policy probability ranks below others."""
    model = _uniform_model()
    candidates = generate_predecessor_candidates(list(successor_tokens), tokenizer=tokenizer, backend="rsshogi")
    assert len(candidates) >= 2

    # Under uniform logits all changed scores tie; the policy stub breaks the tie.
    # Pick the first candidate to be the "bad" one with a tiny policy probability.
    bad_tokens = candidates[0].tokens

    def stub_policy_scorer(successor: tuple[int, ...], cand: list[tuple[tuple[int, ...], str]]) -> list[float]:
        return [1.0e-9 if tokens == bad_tokens else 1.0 for tokens, _ in cand]

    provider = ExpansionProvider(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        candidate_backend="rsshogi",
        per_state_topk=len(candidates),
        policy_scorer=stub_policy_scorer,
        policy_alpha=1.0,
        policy_prefilter_k=len(candidates),
        policy_floor=1.0e-6,
    )
    children = provider.expand_fn(BLEND_MODE)(successor_tokens)
    child_tokens = [tokens for tokens, _ in children]
    assert bad_tokens in child_tokens
    # The low-policy candidate must be ranked last among the returned children.
    assert child_tokens[-1] == bad_tokens


def test_blend_prefilter_limits_policy_evaluation(tokenizer: ShogiTokenizer, successor_tokens: tuple[int, ...]) -> None:
    """Only the top ``policy_prefilter_k`` candidates are passed to the scorer."""
    candidates = generate_predecessor_candidates(list(successor_tokens), tokenizer=tokenizer, backend="rsshogi")
    assert len(candidates) >= 3
    prefilter_k = 2

    # Distinct changed scores so prefiltering is deterministic: logit favoring
    # token value 0 at position 0 makes the ranking depend on candidate tokens.
    logits = torch.zeros((TOTAL_TOKENS, VOCAB_SIZE), dtype=torch.float32)
    model = _ConstantModel(logits)

    evaluated: list[int] = []

    def counting_policy_scorer(successor: tuple[int, ...], cand: list[tuple[tuple[int, ...], str]]) -> list[float]:
        evaluated.append(len(cand))
        return [1.0 for _ in cand]

    provider = ExpansionProvider(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        candidate_backend="rsshogi",
        per_state_topk=len(candidates),
        policy_scorer=counting_policy_scorer,
        policy_alpha=1.0,
        policy_prefilter_k=prefilter_k,
        policy_floor=1.0e-6,
    )
    provider.expand_fn(BLEND_MODE)(successor_tokens)

    assert evaluated == [prefilter_k]


def test_blend_h_threshold_disables_policy_when_far(
    tokenizer: ShogiTokenizer, successor_tokens: tuple[int, ...]
) -> None:
    """Beyond the h threshold the blend mode falls back to pure changed scoring."""
    initial_tokens = tuple(tokenizer.encode(Board()))

    def failing_policy_scorer(successor: tuple[int, ...], cand: list[tuple[tuple[int, ...], str]]) -> list[float]:
        raise AssertionError("policy scorer must not be called beyond the h threshold")

    far_provider = ExpansionProvider(
        model=_uniform_model(),
        tokenizer=tokenizer,
        device="cpu",
        candidate_backend="rsshogi",
        per_state_topk=5,
        policy_scorer=failing_policy_scorer,
        policy_h_threshold=0,
        initial_tokens=initial_tokens,
    )
    # successor_tokens is two plies from the initial position, so h >= 1 > 0.
    children = far_provider.expand_fn(BLEND_MODE)(successor_tokens)
    changed_provider = ExpansionProvider(
        model=_uniform_model(),
        tokenizer=tokenizer,
        device="cpu",
        candidate_backend="rsshogi",
        per_state_topk=5,
    )
    assert children == changed_provider.expand_fn("changed")(successor_tokens)

    called: list[int] = []

    def counting_policy_scorer(successor: tuple[int, ...], cand: list[tuple[tuple[int, ...], str]]) -> list[float]:
        called.append(len(cand))
        return [1.0 for _ in cand]

    near_provider = ExpansionProvider(
        model=_uniform_model(),
        tokenizer=tokenizer,
        device="cpu",
        candidate_backend="rsshogi",
        per_state_topk=5,
        policy_scorer=counting_policy_scorer,
        policy_h_threshold=10,
        initial_tokens=initial_tokens,
    )
    near_provider.expand_fn(BLEND_MODE)(successor_tokens)
    assert called


def test_blend_h_threshold_requires_initial_tokens(tokenizer: ShogiTokenizer) -> None:
    with pytest.raises(ValueError, match="initial_tokens"):
        ExpansionProvider(
            model=_uniform_model(),
            tokenizer=tokenizer,
            device="cpu",
            candidate_backend="rsshogi",
            per_state_topk=5,
            policy_h_threshold=10,
        )


def test_blend_requires_policy_scorer(tokenizer: ShogiTokenizer, successor_tokens: tuple[int, ...]) -> None:
    provider = ExpansionProvider(
        model=_uniform_model(),
        tokenizer=tokenizer,
        device="cpu",
        candidate_backend="rsshogi",
        per_state_topk=5,
    )
    with pytest.raises(ValueError, match="policy scorer"):
        provider.expand_fn(BLEND_MODE)(successor_tokens)


def test_policy_forward_ranks_all_candidates(tokenizer: ShogiTokenizer, successor_tokens: tuple[int, ...]) -> None:
    candidates = generate_predecessor_candidates(list(successor_tokens), tokenizer=tokenizer, backend="rsshogi")
    evaluated: list[int] = []

    def policy_scorer(successor: tuple[int, ...], cand: list[tuple[tuple[int, ...], str]]) -> list[float]:
        evaluated.append(len(cand))
        return [float(index) for index in range(len(cand))]

    provider = ExpansionProvider(
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        candidate_backend="rsshogi",
        per_state_topk=5,
        policy_scorer=policy_scorer,
    )
    children = provider.expand_fn(POLICY_MODE)(successor_tokens)

    assert evaluated == [len(candidates)]
    assert len(children) == min(5, len(candidates))
    assert [score for _, score in children] == sorted((score for _, score in children), reverse=True)
