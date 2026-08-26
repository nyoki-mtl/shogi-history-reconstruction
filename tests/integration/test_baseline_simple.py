"""Simple baseline integration tests for Task-006."""

import time

import numpy as np
from rsshogi.core import Board

from posdiffusion.board.tokenizer import ShogiTokenizer as Tokenizer


class TestBaselineSimple:
    """Simple baseline tests to ensure core functionality works."""

    def test_tokenizer_basic(self):
        """Test basic tokenizer functionality."""
        tokenizer = Tokenizer()

        # Test encoding
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        board = Board(sfen)
        tokens = tokenizer.encode(board)

        assert isinstance(tokens, list)
        assert len(tokens) == 96
        # Convert to numpy array for compatibility
        tokens_array = np.array(tokens, dtype=np.int32)
        assert tokens_array.shape == (96,)

        # Test decoding
        decoded_board = tokenizer.decode(tokens)
        # Compare the SFEN strings
        # SFENの主要部分（盤面、手番、持駒）を比較（手数は保持されない）
        original_parts = sfen.split()
        decoded_parts = decoded_board.to_sfen().split()
        assert decoded_parts[0] == original_parts[0]  # 盤面
        assert decoded_parts[1] == original_parts[1]  # 手番
        assert decoded_parts[2] == original_parts[2]  # 持駒

    def test_performance_baseline(self):
        """Establish performance baseline for tokenizer."""

        tokenizer = Tokenizer()
        n_iterations = 1000

        # Tokenizer encoding performance
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        board = Board(sfen)

        start_time = time.time()
        for _ in range(n_iterations):
            tokens = tokenizer.encode(board)
        encoding_time = time.time() - start_time
        encoding_per_iter = encoding_time / n_iterations

        # Tokenizer decoding performance
        tokens = tokenizer.encode(board)
        start_time = time.time()
        for _ in range(n_iterations):
            tokenizer.decode(tokens)
        decoding_time = time.time() - start_time
        decoding_per_iter = decoding_time / n_iterations

        # Print baseline metrics
        print("\n=== Performance Baseline ===")
        print(f"Tokenizer encoding: {encoding_per_iter * 1000:.3f} ms/iter")
        print(f"Tokenizer decoding: {decoding_per_iter * 1000:.3f} ms/iter")

        # Assert reasonable performance
        assert encoding_per_iter < 0.01  # < 10ms per encoding
        assert decoding_per_iter < 0.01  # < 10ms per decoding
