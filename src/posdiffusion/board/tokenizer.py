"""将棋局面の96トークンエンコーダー/デコーダー。"""

import numpy as np
from rsshogi.core import Board, Move
from rsshogi.types import Color, PieceType, Square

from posdiffusion.board.constants import (
    BOARD_SIZE,
    HAND_PIECE_OFFSET,
    HAND_PIECE_ORDER,
    HAND_SIZE,
    MAX_PAWN_IN_HAND,
    TOTAL_TOKENS,
    TURN_BLACK,
    TURN_WHITE,
)

from .encoding_utils import (
    create_reverse_piece_function,
    hand_piece_id_to_num,
    num_to_hand_piece_id,
)

PIECE_SYMBOLS = ["", "p", "l", "n", "s", "b", "r", "g", "k", "+p", "+l", "+n", "+s", "+b", "+r"]
HAND_PIECE_TYPES = [
    PieceType.PAWN,
    PieceType.LANCE,
    PieceType.KNIGHT,
    PieceType.SILVER,
    PieceType.GOLD,
    PieceType.BISHOP,
    PieceType.ROOK,
]


def _pieces(board: Board) -> list[int]:
    return [int(board.piece_on(Square.from_index(sq))) for sq in range(81)]


def _hands(board: Board) -> tuple[list[int], list[int]]:
    return (
        [board.hand(Color.BLACK).count(piece_type) for piece_type in HAND_PIECE_TYPES],
        [board.hand(Color.WHITE).count(piece_type) for piece_type in HAND_PIECE_TYPES],
    )


class ShogiTokenizer:
    """将棋局面を96トークンで表現するトークナイザー。

    トークン構成:
    - 0-80: 盤面（1一→1二→...→9九）
    - 81-87: 先手持駒（飛、角、金、銀、桂、香、歩）
    - 88-94: 後手持駒（飛、角、金、銀、桂、香、歩）
    - 95: 手番（0=先手、1=後手）
    """

    def __init__(self, max_pawn_in_hand: int = MAX_PAWN_IN_HAND, flip_to_black: bool = False) -> None:
        """トークナイザーを初期化。

        Args:
            max_pawn_in_hand: 歩の最大持駒数（デフォルト: 8）
            flip_to_black: 後手番を先手番に反転するかどうか（デフォルト: False）
        """
        self.max_pawn_in_hand = max_pawn_in_hand
        self.hand_piece_offset = HAND_PIECE_OFFSET
        self.flip_to_black = flip_to_black
        self._reverse_piece_fn = create_reverse_piece_function()

        # Add vocab_size property for compatibility
        self.vocab_size = 40  # Total vocabulary size (0-39: pieces + practical hand pieces max)

    def encode(self, board: Board) -> list[int]:
        """rsshogi Boardオブジェクトを96トークンに変換。

        Args:
            board: 将棋の局面

        Returns:
            96個の整数トークンのリスト
        """
        # 持駒を取得
        black_hand, white_hand = _hands(board)

        # 持駒をトークンIDに変換
        black_hand_tokens = [
            num_to_hand_piece_id(num, self.max_pawn_in_hand, self.hand_piece_offset) for num in black_hand
        ]
        white_hand_tokens = [
            num_to_hand_piece_id(num, self.max_pawn_in_hand, self.hand_piece_offset) for num in white_hand
        ]

        # 手番に応じて盤面をエンコード
        pieces = _pieces(board)
        if board.turn == Color.BLACK or not self.flip_to_black:
            # 先手番または反転無効の場合：そのまま
            tokens = pieces + black_hand_tokens + white_hand_tokens
            # 手番トークンを追加
            turn_token = TURN_BLACK if board.turn == Color.BLACK else TURN_WHITE
            tokens.append(turn_token)
        else:
            # 後手番かつ反転有効の場合：盤面を180度回転し、駒を反転
            reversed_pieces = self._reverse_piece_fn(pieces[::-1])
            tokens = reversed_pieces.tolist() + white_hand_tokens + black_hand_tokens
            # 反転した場合、常に先手番として扱う
            tokens.append(TURN_BLACK)

        return tokens

    def decode(self, tokens: list[int]) -> Board:
        """96トークンをBoardオブジェクトに変換。

        Args:
            tokens: 96個の整数トークンのリスト

        Returns:
            rsshogi Boardオブジェクト
        """
        if len(tokens) != TOTAL_TOKENS:
            raise ValueError(f"Expected {TOTAL_TOKENS} tokens, got {len(tokens)}")

        # トークンを分割
        board_tokens = tokens[:BOARD_SIZE]
        hand_tokens = tokens[BOARD_SIZE : BOARD_SIZE + HAND_SIZE]
        turn_token = tokens[BOARD_SIZE + HAND_SIZE]

        # 盤面部分をSFENに変換
        board_sfen = self._decode_board_to_sfen(board_tokens)

        # 持駒部分をSFENに変換
        hand_sfen = self._decode_hand_to_sfen(hand_tokens)

        # 手番を判定
        turn_char = "b" if turn_token == TURN_BLACK else "w"

        # 結合して完全なSFENを作成
        sfen = f"{board_sfen} {turn_char} {hand_sfen} 1"

        # BoardオブジェクトをSFENから作成
        board = Board()
        board.set_sfen(sfen)
        return board

    def _decode_board_to_sfen(self, board_tokens: list[int]) -> str:
        """盤面トークンをSFEN形式に変換。"""
        sfen_parts = []
        # 1次元配列を9x9に変形し、縦優先から横優先に変換
        board_array = np.array(board_tokens).reshape(9, 9).T[:, ::-1]

        for row in board_array:
            row_sfen = self._decode_row_to_sfen(row)
            sfen_parts.append(row_sfen)

        return "/".join(sfen_parts)

    def _decode_row_to_sfen(self, row: list[int]) -> str:
        """1行分の駒配置をSFEN形式に変換。"""
        row_parts = []
        empty_count = 0

        for piece_id in row:
            # The token representation leaves piece IDs 15 and 16 unused (NOTUSE).
            # piece_id % 16 extracts the piece type (0-15)
            piece_type = piece_id % 16

            # Check if it's an empty square (0) or unused value (15)
            if piece_id == 0 or piece_type == 15:
                # Treat as empty square
                empty_count += 1
            elif piece_type >= len(PIECE_SYMBOLS):
                # This should not happen with valid piece IDs, log warning
                # and treat as empty square to avoid crashes
                import warnings

                warnings.warn(
                    f"Invalid piece type {piece_type} from piece_id {piece_id}, treating as empty", stacklevel=2
                )
                empty_count += 1
            else:
                # Valid piece
                symbol = PIECE_SYMBOLS[piece_type]

                # 先手の駒は大文字
                if piece_id > 0 and piece_id // 16 == 0:
                    symbol = symbol.upper()

                # Add any accumulated empty squares before this piece
                if empty_count > 0:
                    row_parts.append(str(empty_count))
                    empty_count = 0
                row_parts.append(symbol)

        # Add any remaining empty squares at the end of the row
        if empty_count > 0:
            row_parts.append(str(empty_count))

        return "".join(row_parts)

    def _decode_hand_to_sfen(self, hand_tokens: list[int]) -> str:
        """持駒トークンをSFEN形式に変換。"""
        black_hand_tokens = hand_tokens[:7]
        white_hand_tokens = hand_tokens[7:]

        hand_sfen = []

        # 先手持駒
        hand_sfen.extend(self._decode_single_hand(black_hand_tokens, upper=True))
        # 後手持駒
        hand_sfen.extend(self._decode_single_hand(white_hand_tokens, upper=False))

        return "".join(hand_sfen) if hand_sfen else "-"

    def _decode_single_hand(self, hand_tokens: list[int], upper: bool) -> list[str]:
        """片方の持駒をSFEN形式に変換。"""
        result = []
        # SFENの持駒順序は飛角金銀桂香歩なので、逆順にする
        for symbol, token_id in zip(HAND_PIECE_ORDER[::-1], hand_tokens[::-1], strict=False):
            count = hand_piece_id_to_num(token_id, self.max_pawn_in_hand, self.hand_piece_offset)
            if count > 0:
                if count > 1:
                    result.append(str(count))
                result.append(symbol.upper() if upper else symbol)
        return result

    def validate_tokens(self, tokens: list[int]) -> bool:
        """トークンの妥当性を検証。

        Args:
            tokens: 検証するトークンのリスト

        Returns:
            妥当な場合True、そうでない場合False
        """
        if len(tokens) != TOTAL_TOKENS:
            return False

        # 盤面トークンの検証（0-30の範囲）
        for token in tokens[:BOARD_SIZE]:
            if token < 0 or token > 30:
                return False

        # 持駒トークンの検証（31以上）
        for token in tokens[BOARD_SIZE : BOARD_SIZE + HAND_SIZE]:
            if token < self.hand_piece_offset:
                return False

        # 手番トークンの検証（0または1）
        turn_token = tokens[BOARD_SIZE + HAND_SIZE]
        return turn_token in (TURN_BLACK, TURN_WHITE)

    def get_legal_moves(self, board: Board) -> list[Move]:
        """現在の局面の合法手を取得。

        Args:
            board: 将棋の局面

        Returns:
            合法手のリスト
        """
        return board.legal_moves()

    def is_legal_position(self, tokens: list[int]) -> bool:
        """トークンが合法な局面を表しているか検証。

        Args:
            tokens: 検証するトークンのリスト

        Returns:
            合法な局面の場合True、そうでない場合False
        """
        try:
            board = self.decode(tokens)
            # 基本的な検証：両玉が存在するか
            black_king_count = 0
            white_king_count = 0

            for i in range(81):
                piece = int(board.piece_on(Square.from_index(i)))
                if piece == 8:
                    black_king_count += 1
                elif piece == 24:  # 後手の玉
                    white_king_count += 1

            # 両方に玉が1つずつ必要
            # 王手されている側の手番でないことを確認
            # （相手に王手をかけられたまま自分の手番になることはない）
            return black_king_count == 1 and white_king_count == 1

        except (ValueError, IndexError):
            return False
