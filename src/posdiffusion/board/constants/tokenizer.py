"""Constants for tokenization and position encoding."""

from typing import Final

# Token composition
BOARD_SIZE: Final[int] = 81  # 9x9 board
HAND_SIZE: Final[int] = 14  # 7 piece types × 2 players
TURN_SIZE: Final[int] = 1  # Turn indicator
TOTAL_TOKENS: Final[int] = BOARD_SIZE + HAND_SIZE + TURN_SIZE  # 96

# Token index ranges
BOARD_TOKEN_START: Final[int] = 0
BOARD_TOKEN_END: Final[int] = 81
HAND_TOKEN_START: Final[int] = 81
HAND_TOKEN_END: Final[int] = 95
TURN_TOKEN_INDEX: Final[int] = 95

# Hand piece order used by the 96-token representation.
# ["pawn", "lance", "knight", "silver", "gold", "bishop", "rook"]
HAND_PIECE_ORDER: Final[list[str]] = ["p", "l", "n", "s", "g", "b", "r"]

# Hand piece token indices
# Black hand pieces: 81-87
# White hand pieces: 88-94
BLACK_HAND_START: Final[int] = 81
BLACK_HAND_END: Final[int] = 88
WHITE_HAND_START: Final[int] = 88
WHITE_HAND_END: Final[int] = 95

# Turn token values
TURN_BLACK: Final[int] = 0  # Sente
TURN_WHITE: Final[int] = 1  # Gote

# Board dimensions
NUM_RANKS: Final[int] = 9
NUM_FILES: Final[int] = 9

# Hand piece encoding (used for token values 31-39)
HAND_PIECE_OFFSET: Final[int] = 31
MAX_PAWN_IN_HAND: Final[int] = 8  # Maximum count represented by unique tokens

# Piece flip dictionary for color conversion
PIECE_FLIP_DICT: Final[dict[int, int]] = {
    # Black to White
    **{k: k + 16 for k in range(1, 15)},
    # White to Black
    **{k + 16: k for k in range(1, 15)},
    # Empty square
    0: 0,
}
