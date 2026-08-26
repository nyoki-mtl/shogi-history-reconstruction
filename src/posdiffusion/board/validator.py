"""将棋局面の検証ユーティリティ"""

from typing import Any, ClassVar

import numpy as np
from rsshogi.core import Board
from rsshogi.types import Color, PieceType, Square

from posdiffusion.board.constants import (
    BBISHOP,
    BGOLD,
    BKING,
    BKNIGHT,
    BLANCE,
    BPAWN,
    BPROM_BISHOP,
    BPROM_KNIGHT,
    BPROM_LANCE,
    BPROM_PAWN,
    BPROM_ROOK,
    BPROM_SILVER,
    BROOK,
    BSILVER,
    HBISHOP,
    HGOLD,
    HKNIGHT,
    HLANCE,
    HPAWN,
    HROOK,
    HSILVER,
    MAX_PAWN_IN_HAND,
    NONE,
    WBISHOP,
    WGOLD,
    WKING,
    WKNIGHT,
    WLANCE,
    WPAWN,
    WPROM_BISHOP,
    WPROM_KNIGHT,
    WPROM_LANCE,
    WPROM_PAWN,
    WPROM_ROOK,
    WPROM_SILVER,
    WROOK,
    WSILVER,
)
from posdiffusion.board.tokenizer import ShogiTokenizer

HAND_PIECE_TYPES = [
    PieceType.PAWN,
    PieceType.LANCE,
    PieceType.KNIGHT,
    PieceType.SILVER,
    PieceType.GOLD,
    PieceType.BISHOP,
    PieceType.ROOK,
]


def _piece(board: Board, square: int) -> int:
    return int(board.piece_on(Square.from_index(square)))


def _hand(board: Board, color: Color) -> list[int]:
    return [board.hand(color).count(piece_type) for piece_type in HAND_PIECE_TYPES]


class PositionValidator:
    """将棋局面の合法性を検証するクラス"""

    def __init__(self, tokenizer: ShogiTokenizer | None = None) -> None:
        """初期化

        Args:
            tokenizer: 使用するShogiTokenizer。Noneの場合はデフォルトを使用。
        """
        self.tokenizer = tokenizer or ShogiTokenizer()

    # 正しい駒数の定義（先後合わせての個数）
    TOTAL_PIECE_COUNTS: ClassVar[dict[str, int]] = {
        "pawn": 18,  # 歩
        "lance": 4,  # 香
        "knight": 4,  # 桂
        "silver": 4,  # 銀
        "gold": 4,  # 金
        "bishop": 2,  # 角
        "rook": 2,  # 飛
    }

    def is_valid_position(self, board: Board) -> bool:
        """局面の合法性をチェック

        以下の条件を確認します：
        1. 各駒種が先後合わせて正しい個数存在する（成駒は元の駒と合わせてカウント）
        2. 王/玉が各色1個ずつ盤上に存在する

        注意: このメソッドは以下をチェックしません：
        - 二歩などの配置ルール違反
        - 行き所のない駒
        - 王手放置

        特殊ルール：
        - どちらかの持ち歩が8枚（MAX_PAWN_IN_HAND）の場合、歩の合計は17枚以下でもOK
          （トークナイザーが9枚以上の持ち歩を8枚に制限するため）
          この特殊ルールにより、トークナイザーでエンコード・デコードした局面も
          合法と判定されるようになっています。例えば、実際には持ち歩が9枚ある局面を
          トークナイザーでエンコードすると8枚に切り詰められますが、
          その結果の局面も合法として扱われます。

        Args:
            board: チェックするrsshogi Boardオブジェクト

        Returns:
            合法な局面の場合True、そうでない場合False
        """
        # 駒数をカウント
        piece_counts = self._count_pieces(board)

        # 成駒を元の駒種に統合してカウント
        merged_counts = self._merge_promoted_pieces(piece_counts)

        # 王/玉のチェック（必ず各1個、盤上にある必要がある）
        if merged_counts.get(BKING, 0) != 1 or merged_counts.get(WKING, 0) != 1:
            return False

        # 王/玉が盤上にあることを確認
        black_king_on_board = False
        white_king_on_board = False
        for square in range(81):
            piece = _piece(board, square)
            if piece == BKING:
                black_king_on_board = True
            elif piece == WKING:
                white_king_on_board = True

        if not black_king_on_board or not white_king_on_board:
            return False

        # 持ち歩が8枚（最大値）かチェック
        black_hand_pawns = board.hand(Color.BLACK).count(PieceType.PAWN)
        white_hand_pawns = board.hand(Color.WHITE).count(PieceType.PAWN)
        has_max_hand_pawns = MAX_PAWN_IN_HAND in {black_hand_pawns, white_hand_pawns}

        # 各駒種の合計数をチェック
        piece_totals = {
            "pawn": merged_counts.get(BPAWN, 0) + merged_counts.get(WPAWN, 0),
            "lance": merged_counts.get(BLANCE, 0) + merged_counts.get(WLANCE, 0),
            "knight": merged_counts.get(BKNIGHT, 0) + merged_counts.get(WKNIGHT, 0),
            "silver": merged_counts.get(BSILVER, 0) + merged_counts.get(WSILVER, 0),
            "gold": merged_counts.get(BGOLD, 0) + merged_counts.get(WGOLD, 0),
            "bishop": merged_counts.get(BBISHOP, 0) + merged_counts.get(WBISHOP, 0),
            "rook": merged_counts.get(BROOK, 0) + merged_counts.get(WROOK, 0),
        }

        # 各駒種の個数チェック
        for piece_type, expected_count in self.TOTAL_PIECE_COUNTS.items():
            actual_count = piece_totals[piece_type]

            if piece_type == "pawn" and has_max_hand_pawns:
                # 持ち歩が8枚の場合、17枚以下でもOK
                # これは、トークナイザーが9枚の持ち歩を8枚に切り詰めるため、
                # 1枚分の差が生じる可能性があるためです
                if actual_count < expected_count - 1:
                    return False
            # 通常は正確な個数が必要
            elif actual_count != expected_count:
                return False

        # 内部データ構造の整合性もチェック
        return True

    def _count_pieces(self, board: Board) -> dict[int, int]:
        """盤上と持ち駒の全ての駒をカウント

        Args:
            board: カウント対象のBoard

        Returns:
            駒種別の個数辞書
        """
        counts: dict[int, int] = {}

        # 盤上の駒をカウント
        for square in range(81):
            piece = _piece(board, square)
            if piece != NONE:
                counts[piece] = counts.get(piece, 0) + 1

        # 持ち駒をカウント
        for color in [Color.BLACK, Color.WHITE]:
            hand = _hand(board, color)
            for piece_type in range(7):  # PAWN to ROOK
                count = hand[piece_type]
                if count > 0:
                    # 持ち駒の駒種を実際の駒番号に変換
                    piece = self._hand_piece_to_piece(piece_type, color)
                    counts[piece] = counts.get(piece, 0) + count

        return counts

    def _hand_piece_to_piece(self, hand_piece_type: int, color: Color) -> int:
        """持ち駒の駒種を駒番号に変換"""
        piece_map = {
            HPAWN: BPAWN,
            HLANCE: BLANCE,
            HKNIGHT: BKNIGHT,
            HSILVER: BSILVER,
            HGOLD: BGOLD,
            HBISHOP: BBISHOP,
            HROOK: BROOK,
        }
        base_piece = piece_map[hand_piece_type]
        return base_piece if color == Color.BLACK else base_piece + 16

    def _merge_promoted_pieces(self, piece_counts: dict[int, int]) -> dict[int, int]:
        """成駒を元の駒種に統合"""
        merged = piece_counts.copy()

        # 成駒のマッピング
        promoted_map = {
            BPROM_PAWN: BPAWN,
            WPROM_PAWN: WPAWN,
            BPROM_LANCE: BLANCE,
            WPROM_LANCE: WLANCE,
            BPROM_KNIGHT: BKNIGHT,
            WPROM_KNIGHT: WKNIGHT,
            BPROM_SILVER: BSILVER,
            WPROM_SILVER: WSILVER,
            BPROM_BISHOP: BBISHOP,
            WPROM_BISHOP: WBISHOP,
            BPROM_ROOK: BROOK,
            WPROM_ROOK: WROOK,
        }

        # 成駒を元の駒に統合
        for promoted, base in promoted_map.items():
            if promoted in merged:
                merged[base] = merged.get(base, 0) + merged[promoted]
                del merged[promoted]

        return merged

    def _get_base_piece_type(self, piece: int) -> int:
        """駒番号から基本駒種を取得（成駒も元の駒種として扱う）"""
        # 成駒から基本駒種へのマッピング
        promoted_to_base = {
            BPROM_PAWN: BPAWN,
            WPROM_PAWN: WPAWN,
            BPROM_LANCE: BLANCE,
            WPROM_LANCE: WLANCE,
            BPROM_KNIGHT: BKNIGHT,
            WPROM_KNIGHT: WKNIGHT,
            BPROM_SILVER: BSILVER,
            WPROM_SILVER: WSILVER,
            BPROM_BISHOP: BBISHOP,
            WPROM_BISHOP: WBISHOP,
            BPROM_ROOK: BROOK,
            WPROM_ROOK: WROOK,
        }
        return promoted_to_base.get(piece, piece)

    def validate_tokens(self, tokens: np.ndarray[Any, np.dtype[np.int_]] | list[int]) -> tuple[bool, str]:
        """トークン配列の妥当性を検証

        Args:
            tokens: 検証する96個の整数トークン

        Returns:
            (valid, message): 妥当性と検証メッセージのタプル
        """
        tokens_list = tokens.tolist() if isinstance(tokens, np.ndarray) else tokens

        # トークナイザーの検証機能を使用
        if not self.tokenizer.validate_tokens(tokens_list):
            return False, "Invalid token format or values"

        try:
            # Boardオブジェクトに変換してみる
            board = self.tokenizer.decode(tokens_list)

            # 局面の合法性をチェック
            if not self.is_valid_position(board):
                return False, "Invalid position (e.g., double pawn, king in check)"

            return True, "Valid position"

        except Exception as e:
            return False, f"Conversion error: {e}"

    def validate_round_trip(self, board: Board) -> bool:
        """往復変換（Board→tokens→Board）の一致性を検証

        歩がMAX_PAWN_IN_HANDより多い場合は、再変換後にMAX_PAWN_IN_HANDに制限されることを許容する。

        Args:
            board: 検証するrsshogi Boardオブジェクト

        Returns:
            往復変換後に同じ局面（持ち歩制限考慮）になる場合True
        """
        # Board → tokens
        tokens = self.tokenizer.encode(board)

        # tokens → Board
        reconstructed_board = self.tokenizer.decode(tokens)

        # SFEN文字列で比較
        original_sfen = board.to_sfen()
        reconstructed_sfen = reconstructed_board.to_sfen()

        original_parts = original_sfen.split()
        reconstructed_parts = reconstructed_sfen.split()

        if len(original_parts) < 3 or len(reconstructed_parts) < 3:
            return False

        # 盤面と手番は完全一致する必要がある
        if original_parts[0] != reconstructed_parts[0] or original_parts[1] != reconstructed_parts[1]:
            return False

        # 持駒の比較（歩の制限を考慮）
        # SFEN文字列ではなく、内部データで詳細比較
        return self._validate_hand_pieces(board, reconstructed_board)

    def _validate_hand_pieces(self, original_board: Board, reconstructed_board: Board) -> bool:
        """持駒の検証（歩の制限を考慮）"""
        for color in [Color.BLACK, Color.WHITE]:
            original_hand = _hand(original_board, color)
            reconstructed_hand = _hand(reconstructed_board, color)

            for piece_type in range(7):  # PAWN to ROOK
                original_count = original_hand[piece_type]
                reconstructed_count = reconstructed_hand[piece_type]

                if piece_type == HPAWN:
                    # 歩の場合：元が9枚以上なら再変換後は8枚になることを許容
                    if original_count > MAX_PAWN_IN_HAND:
                        if reconstructed_count != MAX_PAWN_IN_HAND:
                            return False
                    elif original_count != reconstructed_count:
                        return False
                # 歩以外は完全一致が必要
                elif original_count != reconstructed_count:
                    return False

        return True
