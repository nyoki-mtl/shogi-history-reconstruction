"""Reverse-generation components.

This package is the home for legal predecessor candidate generation, denoiser
projection scoring, and rollback decoding.
"""

from posdiffusion.reverse.candidates import PredecessorCandidate, generate_predecessor_candidates
from posdiffusion.reverse.scoring import CandidateScores, ScoreMode, score_candidate_tokens

__all__ = [
    "CandidateScores",
    "PredecessorCandidate",
    "ScoreMode",
    "generate_predecessor_candidates",
    "score_candidate_tokens",
]
