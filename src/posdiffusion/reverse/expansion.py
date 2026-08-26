"""Shared expansion provider for legal-projection rollback search.

The provider caches candidate generation and per-mode scoring across search
runs within a single game so that ``best_first_rollback`` and
``iterative_widening_beam_rollback`` can re-expand a state for free.

Two scorer families are supported:

- denoiser token log-probability modes (``full``, ``changed``, ``board``,
  ``board_changed``), non-model controls (``random``, ``rule_order``,
  ``min_token_change``), and the forward-policy-only control
  (``policy_forward``);
- the AlphaZero policy blend mode ``changed_policy_blend``, which ranks all
  candidates by the ``changed`` denoiser score, evaluates AlphaZero policy only
  for the top ``policy_prefilter_k`` candidates, and re-ranks them by a blended
  score ``changed_score + policy_alpha * log(max(policy_prob, policy_floor))``.

With ``policy_h_threshold`` set, blending is applied only when the admissible
distance-to-initial lower bound of the state is at most the threshold.

The policy evaluator is injected as a callable so it can be stubbed in tests.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from rsshogi.core import Move
from rsshogi.policy import move_label

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.diffusion.transformer_denoiser import ShogiTransformerDenoiser
from posdiffusion.evaluation.policy.alphazero_onnx import AlphaZeroONNX
from posdiffusion.reverse.candidates import CandidateBackend, generate_predecessor_candidates
from posdiffusion.reverse.scoring import score_candidate_tokens
from posdiffusion.reverse.search import Tokens, rollback_lower_bound

MODEL_MODES = {"full", "changed", "board", "board_changed"}
CONTROL_MODES = {"random", "rule_order", "min_token_change"}
BLEND_MODE = "changed_policy_blend"
POLICY_MODE = "policy_forward"
ALL_MODES = MODEL_MODES | CONTROL_MODES | {BLEND_MODE, POLICY_MODE}

# A policy scorer receives the successor tokens and a list of (predecessor
# tokens, forward move USI) candidates, and returns one normalized forward-move
# probability per candidate.
PolicyScorer = Callable[[Tokens, list[tuple[Tokens, str]]], list[float]]


@dataclass(frozen=True)
class Candidate:
    """A legal predecessor candidate: its tokens and the forward move USI."""

    tokens: Tokens
    move_usi: str


def deterministic_random_score(*, seed: int, source_tokens: Tokens, candidate_tokens: Tokens) -> float:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(seed).encode("ascii"))
    digest.update(",".join(str(token) for token in source_tokens).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(token) for token in candidate_tokens).encode("ascii"))
    return int.from_bytes(digest.digest(), byteorder="big") / float(1 << 64)


def make_alphazero_policy_scorer(alphazero: AlphaZeroONNX, tokenizer: ShogiTokenizer) -> PolicyScorer:
    """Build a policy scorer backed by an AlphaZero ONNX model.

    Each candidate predecessor board is decoded and scored with the policy
    head.  The forward move probability is normalized by the legal-move
    probability mass of that predecessor board, matching the normalization used
    by the ``policy_forward`` mode of the beam rollback evaluator.
    """

    def score(successor_tokens: Tokens, candidates: list[tuple[Tokens, str]]) -> list[float]:
        if not candidates:
            return []
        predecessor_boards = [tokenizer.decode(list(tokens)) for tokens, _ in candidates]
        policy_probs_batch, _ = alphazero.get_policy_value_batch(predecessor_boards)
        probabilities: list[float] = []
        for (_, move_usi), predecessor_board, policy_probs in zip(
            candidates, predecessor_boards, policy_probs_batch, strict=True
        ):
            legal_total = 0.0
            for legal_move in predecessor_board.legal_moves():
                legal_total += float(policy_probs[move_label(legal_move, predecessor_board.turn)])
            move = Move.from_usi(move_usi)
            probability = float(policy_probs[move_label(move, predecessor_board.turn)])
            if legal_total > 0.0:
                probability /= legal_total
            probabilities.append(probability)
        return probabilities

    return score


class ExpansionProvider:
    """Cache candidate generation and scoring within one search session."""

    def __init__(
        self,
        *,
        model: ShogiTransformerDenoiser | None,
        tokenizer: ShogiTokenizer,
        device: str,
        candidate_backend: CandidateBackend,
        per_state_topk: int,
        random_seed: int = 0,
        policy_scorer: PolicyScorer | None = None,
        policy_alpha: float = 1.0,
        policy_prefilter_k: int = 30,
        policy_floor: float = 1.0e-6,
        policy_h_threshold: int | None = None,
        initial_tokens: Tokens | None = None,
    ) -> None:
        if per_state_topk <= 0:
            raise ValueError("per_state_topk must be positive")
        if policy_prefilter_k <= 0:
            raise ValueError("policy_prefilter_k must be positive")
        if policy_floor <= 0.0:
            raise ValueError("policy_floor must be positive")
        if policy_h_threshold is not None:
            if policy_h_threshold < 0:
                raise ValueError("policy_h_threshold must be non-negative")
            if initial_tokens is None:
                raise ValueError("policy_h_threshold requires initial_tokens")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.candidate_backend = candidate_backend
        self.per_state_topk = per_state_topk
        self.random_seed = random_seed
        self.policy_scorer = policy_scorer
        self.policy_alpha = policy_alpha
        self.policy_prefilter_k = policy_prefilter_k
        self.policy_floor = policy_floor
        self.policy_h_threshold = policy_h_threshold
        self.initial_tokens = initial_tokens
        self.candidate_cache: dict[Tokens, list[Candidate]] = {}
        self.children_cache: dict[tuple[Tokens, str], list[tuple[Tokens, float]]] = {}

    def _candidates(self, tokens: Tokens) -> list[Candidate]:
        cached = self.candidate_cache.get(tokens)
        if cached is None:
            generated = generate_predecessor_candidates(
                list(tokens), tokenizer=self.tokenizer, backend=self.candidate_backend
            )
            cached = [Candidate(tokens=candidate.tokens, move_usi=candidate.move_usi) for candidate in generated]
            self.candidate_cache[tokens] = cached
        return cached

    def _denoiser_scores(self, tokens: Tokens, candidate_tokens: list[Tokens], mode: str) -> list[float]:
        if self.model is None:
            raise ValueError(f"Mode {mode} requires --checkpoint")
        source = torch.tensor(tokens, dtype=torch.long, device=self.device)
        candidates = torch.tensor(candidate_tokens, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(
                source.unsqueeze(0),
                torch.ones(1, device=self.device, dtype=torch.long),
                absolute_move=None,
                game_phase=None,
            )[0]
        scores = score_candidate_tokens(logits, source, candidates, mode=mode).scores  # type: ignore[arg-type]
        return [float(score) for score in scores.detach().cpu().tolist()]

    def _scores(self, tokens: Tokens, candidate_tokens: list[Tokens], mode: str) -> list[float]:
        if mode in MODEL_MODES:
            return self._denoiser_scores(tokens, candidate_tokens, mode)
        if mode == "random":
            return [
                deterministic_random_score(seed=self.random_seed, source_tokens=tokens, candidate_tokens=candidate)
                for candidate in candidate_tokens
            ]
        if mode == "rule_order":
            return [-float(index) for index in range(len(candidate_tokens))]
        if mode == "min_token_change":
            return [
                -float(sum(a != b for a, b in zip(candidate, tokens, strict=True))) - index * 1.0e-9
                for index, candidate in enumerate(candidate_tokens)
            ]
        raise ValueError(f"Unknown score mode: {mode}")

    def _blend_children(self, tokens: Tokens, candidates: list[Candidate]) -> list[tuple[Tokens, float]]:
        if self.policy_scorer is None:
            raise ValueError(f"Mode {BLEND_MODE} requires a policy scorer")
        candidate_tokens = [candidate.tokens for candidate in candidates]
        changed_scores = self._denoiser_scores(tokens, candidate_tokens, "changed")
        ranked = sorted(zip(changed_scores, candidates, strict=True), key=lambda item: item[0], reverse=True)
        prefilter = ranked[: self.policy_prefilter_k]
        policy_inputs = [(candidate.tokens, candidate.move_usi) for _, candidate in prefilter]
        policy_probs = self.policy_scorer(tokens, policy_inputs)
        blended = sorted(
            (
                (changed_score + self.policy_alpha * math.log(max(policy_prob, self.policy_floor)), candidate.tokens)
                for (changed_score, candidate), policy_prob in zip(prefilter, policy_probs, strict=True)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [(child_tokens, score) for score, child_tokens in blended[: self.per_state_topk]]

    def _policy_children(self, tokens: Tokens, candidates: list[Candidate]) -> list[tuple[Tokens, float]]:
        if self.policy_scorer is None:
            raise ValueError(f"Mode {POLICY_MODE} requires a policy scorer")
        policy_inputs = [(candidate.tokens, candidate.move_usi) for candidate in candidates]
        probabilities = self.policy_scorer(tokens, policy_inputs)
        ranked = sorted(
            zip(probabilities, candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
        return [(candidate.tokens, probability) for probability, candidate in ranked[: self.per_state_topk]]

    def expand_fn(self, mode: str) -> Callable[[Tokens], list[tuple[Tokens, float]]]:
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown score mode: {mode}")

        def expand(tokens: Tokens) -> list[tuple[Tokens, float]]:
            key = (tokens, mode)
            cached = self.children_cache.get(key)
            if cached is not None:
                return cached
            candidates = self._candidates(tokens)
            if not candidates:
                self.children_cache[key] = []
                return []
            blend_active = mode == BLEND_MODE and (
                self.policy_h_threshold is None
                or rollback_lower_bound(tokens, self.initial_tokens) <= self.policy_h_threshold  # type: ignore[arg-type]
            )
            if mode == POLICY_MODE:
                children = self._policy_children(tokens, candidates)
            elif mode == BLEND_MODE and not blend_active:
                candidate_tokens = [candidate.tokens for candidate in candidates]
                scores = self._denoiser_scores(tokens, candidate_tokens, "changed")
                ranked = sorted(zip(scores, candidate_tokens, strict=True), key=lambda item: item[0], reverse=True)
                children = [(candidate, score) for score, candidate in ranked[: self.per_state_topk]]
            elif mode == BLEND_MODE:
                children = self._blend_children(tokens, candidates)
            else:
                candidate_tokens = [candidate.tokens for candidate in candidates]
                scores = self._scores(tokens, candidate_tokens, mode)
                ranked = sorted(zip(scores, candidate_tokens, strict=True), key=lambda item: item[0], reverse=True)
                children = [(candidate, score) for score, candidate in ranked[: self.per_state_topk]]
            self.children_cache[key] = children
            return children

        return expand
