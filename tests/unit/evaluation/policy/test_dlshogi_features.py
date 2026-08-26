"""Regression tests for rsshogi-based dlshogi feature extraction."""

import hashlib

import numpy as np
import pytest
from rsshogi.core import Board

from posdiffusion.evaluation.policy.dlshogi_features import (
    FEATURES1_NUM,
    FEATURES2_NUM,
    make_input_features,
)

# (sfen, features1 sha256[:16], features2 sha256[:16]) reference hashes.
GOLDEN_CASES: list[tuple[str, str, str]] = [
    ("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", "3c58e5027f56b73b", "482fd764aff74b78"),
    ("lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2", "8f13d14c3ca9e61f", "482fd764aff74b78"),
    (
        "ln1g5/1ks4+Bl/1pp+R3p1/p4p2p/3p1S3/P1P5P/1P1G1P3/1KSG3+r1/LN6L w BG2N2Pgsp 1",
        "1514ce27ba4d3663",
        "f43bda6f7e4c13be",
    ),
    ("lnsgkg1nl/1r5s1/pppppp1pp/6p2/4B4/2P6/PP1PPPPPP/7R1/LNSGKGSNL w b 6", "7a431662086ef101", "9d549b91ed3a4d11"),
    (
        "1nsgkgsn1/lr6l/1p1ppp1pp/p1p3p2/9/4P2P1/PP+bP1PP1P/LB2R1S2/1NSGKG1NL b p 15",
        "0dd733c56a98f28f",
        "d178729aa6006c90",
    ),  # black to move, in check
]


def _make_features(sfen: str) -> tuple[np.ndarray, np.ndarray]:
    board = Board(sfen)
    x1 = np.empty((FEATURES1_NUM, 9, 9), dtype=np.float32)
    x2 = np.empty((FEATURES2_NUM, 9, 9), dtype=np.float32)
    make_input_features(board, x1, x2)
    return x1, x2


def test_feature_dimensions() -> None:
    assert FEATURES1_NUM == 62
    assert FEATURES2_NUM == 57


@pytest.mark.parametrize(("sfen", "hash1", "hash2"), GOLDEN_CASES)
def test_features_match_reference_golden(sfen: str, hash1: str, hash2: str) -> None:
    x1, x2 = _make_features(sfen)
    assert hashlib.sha256(x1.tobytes()).hexdigest()[:16] == hash1
    assert hashlib.sha256(x2.tobytes()).hexdigest()[:16] == hash2


def test_features_are_binary() -> None:
    x1, x2 = _make_features("ln1g5/1ks4+Bl/1pp+R3p1/p4p2p/3p1S3/P1P5P/1P1G1P3/1KSG3+r1/LN6L w BG2N2Pgsp 1")
    assert set(np.unique(x1)).issubset({0.0, 1.0})
    assert set(np.unique(x2)).issubset({0.0, 1.0})


def test_check_channel_set_when_in_check() -> None:
    board = Board("1nsgkgsn1/lr6l/1p1ppp1pp/p1p3p2/9/4P2P1/PP+bP1PP1P/LB2R1S2/1NSGKG1NL b p 15")
    assert board.is_in_check()
    _, x2 = _make_features(board.to_sfen())
    # Last channel (index FEATURES2_NUM - 1) is the check indicator plane.
    assert np.all(x2[FEATURES2_NUM - 1] == 1.0)


def test_check_channel_clear_when_not_in_check() -> None:
    board = Board("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    assert not board.is_in_check()
    _, x2 = _make_features(board.to_sfen())
    assert np.all(x2[FEATURES2_NUM - 1] == 0.0)
