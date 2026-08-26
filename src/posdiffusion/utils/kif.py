"""KIF parser for extracting moves from KIF files."""

import re
from pathlib import Path
from typing import Any

from rsshogi.core import Board, Move
from rsshogi.types import Color, Square

PIECE_SYMBOLS = ["", "p", "l", "n", "s", "b", "r", "g", "k", "+p", "+l", "+n", "+s", "+b", "+r"]
PIECE_JAPANESE_SYMBOLS = ["", "歩", "香", "桂", "銀", "角", "飛", "金", "玉", "と", "杏", "圭", "全", "馬", "龍"]
NUMBER_JAPANESE_NUMBER_SYMBOLS = ["", "１", "２", "３", "４", "５", "６", "７", "８", "９"]
NUMBER_JAPANESE_KANJI_SYMBOLS = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
SQUARE_NAMES = [Square.from_index(i).to_usi() for i in range(81)]


def _winner_from_turn(turn: Color, *, opponent: bool) -> int:
    turn_as_int = 0 if turn == Color.BLACK else 1
    return 2 - turn_as_int if opponent else 1 + turn_as_int


# Simplified KIF parser
MOVE_PTN = re.compile(
    r"\A *[0-9]+ (中断|投了|持将棋|千日手|詰み|切れ負け|反則勝ち|反則負け|入玉宣言|入玉勝ち|"
    + r"(([１２３４５６７８９])([零一二三四五六七八九])|同　)([歩香桂銀金角飛玉と杏圭全馬龍])"
    + r"(打|(成?)\(([0-9])([0-9])\)))\s*(\(([ :0-9]+)/([ :0-9]+)\))?\s*\Z"
)

RESULT_LIST = ["中断", "投了", "持将棋", "千日手", "詰み", "切れ負け", "反則勝ち", "反則負け", "入玉宣言", "入玉勝ち"]


class KIFParser:
    """Simple KIF parser for extracting moves."""

    def __init__(self) -> None:
        self.board = Board()

    def parse_kif_file(self, kif_path: Path, encoding: str = "shift_jis") -> dict[str, Any] | None:
        """Parse a KIF file and extract moves."""
        kif_str = kif_path.read_text(encoding=encoding)
        return self.parse_kif_str(kif_str)

    def parse_kif_str(self, kif_str: str) -> dict[str, Any]:
        """Parse KIF string and extract game information."""
        lines = kif_str.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        moves = []
        endgame = ""
        win = -1
        last_to_square = None

        self.board.reset()

        for line in lines:
            move_line = self._normalize_move_line(line)
            m = MOVE_PTN.match(move_line)

            if m is not None:
                if endgame != "":
                    continue

                if m.group(1) not in RESULT_LIST:
                    move, last_to_square = self._parse_regular_move(m, last_to_square)
                    if move is None:
                        endgame = "%ILLEGAL_MOVE"
                        win = _winner_from_turn(self.board.turn, opponent=True)
                        break
                    moves.append(move)
                    self.board.push_move(move)
                else:
                    endgame, win = self._parse_game_end(m.group(1))

        if win == -1:
            win = 0
            endgame = "%CHUDAN"

        return {
            "moves": moves,
            "endgame": endgame,
            "win": win,
        }

    def _normalize_move_line(self, line: str) -> str:
        """Normalize move line by replacing characters."""
        move_line = line
        move_line = move_line.replace("王", "玉")
        move_line = move_line.replace("竜", "龍")
        move_line = move_line.replace("成銀", "全")
        move_line = move_line.replace("成桂", "圭")
        move_line = move_line.replace("成香", "杏")
        return move_line

    def _parse_regular_move(self, match: re.Match[str], last_to_square: int | None) -> tuple[Move | None, int | None]:
        """Parse a regular move from regex match."""
        piece_type = PIECE_JAPANESE_SYMBOLS.index(match.group(5))

        if match.group(2) == "同　":
            assert last_to_square is not None, "last_to_square is not set"
            to_square = last_to_square
        else:
            to_field = NUMBER_JAPANESE_NUMBER_SYMBOLS.index(match.group(3)) - 1
            to_rank = NUMBER_JAPANESE_KANJI_SYMBOLS.index(match.group(4)) - 1
            to_square = to_rank + to_field * 9
            last_to_square = to_square

        if match.group(6) == "打":
            move_usi = f"{PIECE_SYMBOLS[piece_type].upper()}*{SQUARE_NAMES[to_square]}"
        else:
            from_field = int(match.group(8)) - 1
            from_rank = int(match.group(9)) - 1
            from_square = from_rank + from_field * 9

            promotion = match.group(7) == "成"
            move_usi = SQUARE_NAMES[from_square] + SQUARE_NAMES[to_square] + ("+" if promotion else "")

        move = Move.from_usi(move_usi)
        if not self.board.is_legal_move(move):
            return None, last_to_square

        return move, last_to_square

    def _parse_game_end(self, endgame_ja: str) -> tuple[str, int]:
        """Parse game end result."""
        endgame_mapping = {
            "投了": ("%TORYO", _winner_from_turn(self.board.turn, opponent=True)),
            "詰み": ("%TORYO", _winner_from_turn(self.board.turn, opponent=True)),
            "持将棋": ("%JISHOGI", 0),
            "千日手": ("%SENNICHITE", 0),
            "切れ負け": ("%TIME_UP", _winner_from_turn(self.board.turn, opponent=True)),
            "反則勝ち": ("%ILLEGAL_MOVE", _winner_from_turn(self.board.turn, opponent=False)),
            "反則負け": ("%ILLEGAL_MOVE", _winner_from_turn(self.board.turn, opponent=True)),
            "中断": ("%CHUDAN", 0),
            "入玉宣言": ("%KACHI", _winner_from_turn(self.board.turn, opponent=False)),
            "入玉勝ち": ("%KACHI", _winner_from_turn(self.board.turn, opponent=False)),
        }

        return endgame_mapping.get(endgame_ja, ("%CHUDAN", 0))
