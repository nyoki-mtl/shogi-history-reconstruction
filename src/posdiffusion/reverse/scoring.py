"""Scoring utilities for legal predecessor projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from posdiffusion.board.constants.tokenizer import BOARD_TOKEN_END, HAND_TOKEN_END, HAND_TOKEN_START, TOTAL_TOKENS
from posdiffusion.reverse.candidates import PredecessorCandidate

ScoreMode = Literal["full", "changed", "board", "hand", "turn", "board_changed", "hand_changed"]


@dataclass(frozen=True)
class CandidateScores:
    """Scores for one candidate set under one scoring mode."""

    mode: ScoreMode
    scores: torch.Tensor


def positions_for_mode(
    *,
    mode: ScoreMode,
    source_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return a boolean mask over token positions used by a scoring mode."""

    if source_tokens.shape != (TOTAL_TOKENS,):
        raise ValueError(f"source_tokens must have shape ({TOTAL_TOKENS},)")
    if candidate_tokens.ndim != 2 or candidate_tokens.shape[1] != TOTAL_TOKENS:
        raise ValueError(f"candidate_tokens must have shape (num_candidates, {TOTAL_TOKENS})")

    device = candidate_tokens.device
    base = torch.zeros((candidate_tokens.shape[0], TOTAL_TOKENS), dtype=torch.bool, device=device)
    if mode == "full":
        base[:, :] = True
    elif mode == "changed":
        base = candidate_tokens != source_tokens.unsqueeze(0)
    elif mode == "board":
        base[:, :BOARD_TOKEN_END] = True
    elif mode == "hand":
        base[:, HAND_TOKEN_START:HAND_TOKEN_END] = True
    elif mode == "turn":
        base[:, HAND_TOKEN_END] = True
    elif mode == "board_changed":
        base[:, :BOARD_TOKEN_END] = candidate_tokens[:, :BOARD_TOKEN_END] != source_tokens[:BOARD_TOKEN_END]
    elif mode == "hand_changed":
        base[:, HAND_TOKEN_START:HAND_TOKEN_END] = (
            candidate_tokens[:, HAND_TOKEN_START:HAND_TOKEN_END] != source_tokens[HAND_TOKEN_START:HAND_TOKEN_END]
        )
    else:
        raise ValueError(f"Unsupported score mode: {mode}")
    return base


def score_candidate_tokens(
    logits: torch.Tensor,
    source_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    *,
    mode: ScoreMode,
) -> CandidateScores:
    """Score candidate predecessor tokens using denoiser token log-probabilities."""

    if logits.shape[0] != TOTAL_TOKENS:
        raise ValueError(f"logits must have shape ({TOTAL_TOKENS}, vocab_size)")
    if candidate_tokens.numel() == 0:
        return CandidateScores(mode=mode, scores=torch.empty(0, dtype=logits.dtype, device=logits.device))

    log_probs = torch.log_softmax(logits, dim=-1)
    position_ids = torch.arange(TOTAL_TOKENS, device=logits.device).unsqueeze(0).expand(candidate_tokens.shape[0], -1)
    token_scores = log_probs[position_ids, candidate_tokens]
    mask = positions_for_mode(mode=mode, source_tokens=source_tokens, candidate_tokens=candidate_tokens)
    return CandidateScores(mode=mode, scores=(token_scores * mask.to(token_scores.dtype)).sum(dim=1))


def candidates_to_tensor(candidates: list[PredecessorCandidate], *, device: torch.device | str) -> torch.Tensor:
    """Convert predecessor candidates to a long tensor with shape (N, 96)."""

    return torch.tensor([candidate.tokens for candidate in candidates], dtype=torch.long, device=device)


def true_candidate_index(candidates: list[PredecessorCandidate], true_tokens: tuple[int, ...]) -> int | None:
    """Return the true predecessor index in candidate order, or None if uncovered."""

    for idx, candidate in enumerate(candidates):
        if candidate.tokens == true_tokens:
            return idx
    return None
