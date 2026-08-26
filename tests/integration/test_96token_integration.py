"""96トークン実装の統合テスト。"""

import pytest
from rsshogi.core import Board

from posdiffusion.board import ShogiTokenizer
from posdiffusion.board.constants import TOTAL_TOKENS, TURN_BLACK, TURN_WHITE


class Test96TokenIntegration:
    """96トークン実装の統合テスト。"""

    @pytest.fixture
    def tokenizer(self) -> ShogiTokenizer:
        """テスト用のトークナイザー。"""
        return ShogiTokenizer()

    def test_token_count(self, tokenizer: ShogiTokenizer) -> None:
        """トークン数が96であることを確認。"""
        assert TOTAL_TOKENS == 96

    def test_different_turns_different_tokens(self, tokenizer: ShogiTokenizer) -> None:
        """同じ盤面でも手番が異なれば異なるトークンになることを確認。"""
        # 初期局面から1手進めた局面を作成
        board = Board()
        board.push_usi("7g7f")

        # 後手番の状態でエンコード
        tokens_white_turn = tokenizer.encode(board)

        # 同じ盤面だが先手番にする
        board_sfen = board.to_sfen()
        parts = board_sfen.split()
        parts[1] = "b"  # 手番を先手に変更
        modified_sfen = " ".join(parts)

        board_black_turn = Board()
        board_black_turn.set_sfen(modified_sfen)
        tokens_black_turn = tokenizer.encode(board_black_turn)

        # 手番トークンが異なることを確認
        assert tokens_white_turn[95] == TURN_WHITE
        assert tokens_black_turn[95] == TURN_BLACK

        # 盤面と持駒は同じであることを確認
        assert tokens_white_turn[:95] == tokens_black_turn[:95]

    def test_decode_preserves_turn(self, tokenizer: ShogiTokenizer) -> None:
        """デコード時に手番が正しく保持されることを確認。"""
        # いくつかの手を進めた局面
        board = Board()
        moves = ["7g7f", "3c3d", "2g2f", "8c8d", "2f2e"]
        for move in moves:
            board.push_usi(move)

        # エンコード→デコード
        tokens = tokenizer.encode(board)
        decoded_board = tokenizer.decode(tokens)

        # 手番が正しく保持されていることを確認
        assert board.turn == decoded_board.turn

        # SFEN表現の重要部分（盤面、手番、持駒）が一致することを確認
        # 手数は保持されないので比較しない
        original_parts = board.to_sfen().split()
        decoded_parts = decoded_board.to_sfen().split()

        assert original_parts[0] == decoded_parts[0]  # 盤面
        assert original_parts[1] == decoded_parts[1]  # 手番
        assert original_parts[2] == decoded_parts[2]  # 持駒

    def test_legal_moves_depend_on_turn(self, tokenizer: ShogiTokenizer) -> None:
        """合法手が手番によって異なることを確認。"""
        # 初期局面
        board = Board()

        # 先手番の合法手
        black_legal_moves = board.legal_moves()

        # 同じ盤面で後手番にする
        board_sfen = board.to_sfen()
        parts = board_sfen.split()
        parts[1] = "w"  # 手番を後手に変更
        modified_sfen = " ".join(parts)

        board_white = Board()
        board_white.set_sfen(modified_sfen)

        # 後手番の合法手
        white_legal_moves = board_white.legal_moves()

        # 合法手が異なることを確認（初期局面では先手と後手で全く異なる手）
        assert len(black_legal_moves) > 0
        assert len(white_legal_moves) > 0
        assert all(move not in white_legal_moves for move in black_legal_moves)
