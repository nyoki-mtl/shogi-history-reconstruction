"""ShogiTokenizerの単体テスト。"""

import pytest
from rsshogi.core import Board
from rsshogi.types import Square

from posdiffusion.board import ShogiTokenizer
from posdiffusion.board.constants import TOTAL_TOKENS


def _pieces(board: Board) -> list[int]:
    return [int(board.piece_on(Square.from_index(sq))) for sq in range(81)]


class TestShogiTokenizer:
    """ShogiTokenizerのテストクラス。"""

    @pytest.fixture
    def tokenizer(self) -> ShogiTokenizer:
        """テスト用のトークナイザーインスタンス。"""
        return ShogiTokenizer()

    def test_encode_initial_position(self, tokenizer: ShogiTokenizer) -> None:
        """初期局面のエンコードテスト。"""
        board = Board()
        tokens = tokenizer.encode(board)

        # トークン数の確認
        assert len(tokens) == TOTAL_TOKENS

        # 初期局面の特定の駒を確認
        # 1一の香車（後手）
        assert tokens[0] == 18  # W_LANCE
        # 5五の空きマス
        assert tokens[40] == 0  # EMPTY
        # 9九の香車（先手）
        assert tokens[80] == 2  # B_LANCE

        # 持駒は全て0枚（ID: 31）
        for i in range(81, 95):
            assert tokens[i] == 31

        # 手番は先手（0）
        assert tokens[95] == 0

    def test_encode_with_captured_pieces(self, tokenizer: ShogiTokenizer) -> None:
        """持駒がある局面のエンコードテスト。"""
        board = Board()
        # 適当な手を指して持駒を作る
        board.push_usi("7g7f")
        board.push_usi("3c3d")
        board.push_usi("8h2b+")  # 角を取る
        board.push_usi("3a2b")

        tokens = tokenizer.encode(board)

        # 先手が角を1枚持っている
        # 角は持駒順序で6番目（歩、香、桂、銀、金、角、飛）
        assert tokens[86] == 32  # 1枚

        # 手番は先手（0）
        assert tokens[95] == 0

    def test_encode_white_turn(self, tokenizer: ShogiTokenizer) -> None:
        """後手番の局面のエンコードテスト。"""
        board = Board()
        board.push_usi("7g7f")  # 先手の手

        # 後手番の状態でエンコード
        tokens = tokenizer.encode(board)

        # 盤面が180度回転し、駒が反転されているか確認
        # 9九の位置（インデックス80）には3三の香（先手から見て）
        assert tokens[80] == 2  # B_LANCE（反転済み）

        # 手番は後手（1）
        assert tokens[95] == 1

    def test_decode_initial_position(self, tokenizer: ShogiTokenizer) -> None:
        """初期局面のデコードテスト。"""
        board = Board()
        tokens = tokenizer.encode(board)
        decoded_board = tokenizer.decode(tokens)

        # 初期局面のSFEN（手番情報も含めて正確に復元される）
        expected = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        assert decoded_board.to_sfen() == expected

    def test_roundtrip_conversion(self, tokenizer: ShogiTokenizer) -> None:
        """エンコード→デコードのラウンドトリップテスト。"""
        # いくつかの手を指した局面
        board = Board()
        board.push_usi("7g7f")
        board.push_usi("3c3d")
        board.push_usi("2g2f")
        board.push_usi("8c8d")

        # エンコード→デコード
        tokens = tokenizer.encode(board)
        decoded_board = tokenizer.decode(tokens)

        # デコードしたBoardオブジェクトを使用
        new_board = decoded_board

        # 元の局面と同じ盤面か確認
        assert _pieces(board) == _pieces(new_board)
        # 手番も正しく復元される
        assert board.turn == new_board.turn

    def test_validate_tokens_valid(self, tokenizer: ShogiTokenizer) -> None:
        """正常なトークンの検証テスト。"""
        board = Board()
        tokens = tokenizer.encode(board)

        assert tokenizer.validate_tokens(tokens) is True

    def test_validate_tokens_invalid_length(self, tokenizer: ShogiTokenizer) -> None:
        """不正な長さのトークンの検証テスト。"""
        tokens = [0] * 94  # 1つ足りない
        assert tokenizer.validate_tokens(tokens) is False

        tokens = [0] * 97  # 1つ多い
        assert tokenizer.validate_tokens(tokens) is False

    def test_validate_tokens_invalid_values(self, tokenizer: ShogiTokenizer) -> None:
        """不正な値を含むトークンの検証テスト。"""
        tokens = [0] * 96  # 96トークンに変更

        # 盤面部分に不正な値
        tokens[0] = -1
        assert tokenizer.validate_tokens(tokens) is False

        tokens[0] = 31  # 盤面に持駒ID
        assert tokenizer.validate_tokens(tokens) is False

        # 持駒部分に不正な値
        tokens = [0] * 96  # 96トークンに変更
        tokens[81] = 30  # 持駒部分に駒ID
        assert tokenizer.validate_tokens(tokens) is False

        # 手番トークンに不正な値
        tokens = [0] * 81 + [31] * 14 + [2]  # 手番に2（無効）
        assert tokenizer.validate_tokens(tokens) is False

    def test_promoted_pieces(self, tokenizer: ShogiTokenizer) -> None:
        """成駒のエンコード/デコードテスト。"""
        # 成駒を取った時のテスト（持駒は成っていない状態になる）
        board = Board()
        board.push_usi("7g7f")
        board.push_usi("3c3d")
        board.push_usi("8h2b+")  # 角を成って取る
        board.push_usi("3a2b")  # 成角を取る

        tokens = tokenizer.encode(board)

        # 両者が角を1枚ずつ持っている（成駒を取っても持駒では成っていない）
        # 角は持駒順序で6番目（歩、香、桂、銀、金、角、飛）
        assert tokens[86] == 32  # 先手の角1枚
        assert tokens[93] == 32  # 後手の角1枚

    def test_max_pawns_in_hand(self, tokenizer: ShogiTokenizer) -> None:
        """歩の最大持駒数のテスト。"""
        # 歩を8枚持っている局面のSFEN
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b 8P 1"
        board = Board(sfen)

        tokens = tokenizer.encode(board)

        # 歩は最初（インデックス81）
        assert tokens[81] == 39  # 31 + 8 = 39
