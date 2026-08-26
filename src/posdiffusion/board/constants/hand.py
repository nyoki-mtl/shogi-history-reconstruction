"""Hand piece constants and utilities."""

from typing import Final

from .base import (
    BBISHOP,
    BGOLD,
    BKNIGHT,
    BLACK,
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
    WBISHOP,
    WGOLD,
    WHITE,
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

# Target piece counts for constraint calculations
# Maximum possible pieces that can be captured
TARGET_PIECE_COUNTS: Final[dict[str, float]] = {
    "pawn": 18.0,  # 9 pawns per side, but can have up to 18 through promotion
    "lance": 4.0,  # 2 lances per side
    "knight": 4.0,  # 2 knights per side
    "silver": 4.0,  # 2 silvers per side
    "gold": 4.0,  # 2 golds per side
    "bishop": 2.0,  # 1 bishop per side
    "rook": 2.0,  # 1 rook per side
}


def generate_piece_mappings() -> dict[int, int]:
    """Generate mapping from piece ID to piece type (0-6).

    Piece types: 0:pawn, 1:lance, 2:knight, 3:silver, 4:gold, 5:bishop, 6:rook.
    """
    return {
        # Black
        BPAWN: 0,
        BPROM_PAWN: 0,
        BLANCE: 1,
        BPROM_LANCE: 1,
        BKNIGHT: 2,
        BPROM_KNIGHT: 2,
        BSILVER: 3,
        BPROM_SILVER: 3,
        BBISHOP: 5,
        BPROM_BISHOP: 5,
        BROOK: 6,
        BPROM_ROOK: 6,
        BGOLD: 4,
        # White
        WPAWN: 0,
        WPROM_PAWN: 0,
        WLANCE: 1,
        WPROM_LANCE: 1,
        WKNIGHT: 2,
        WPROM_KNIGHT: 2,
        WSILVER: 3,
        WPROM_SILVER: 3,
        WBISHOP: 5,
        WPROM_BISHOP: 5,
        WROOK: 6,
        WPROM_ROOK: 6,
        WGOLD: 4,
    }


# Hand piece token indices
# Black hand pieces: tokens 81-87
# White hand pieces: tokens 88-94
HAND_PIECE_INDICES: Final[dict[int, dict[int, int]]] = {
    BLACK: {
        0: 81,  # HPAWN
        1: 82,  # HLANCE
        2: 83,  # HKNIGHT
        3: 84,  # HSILVER
        4: 85,  # HGOLD
        5: 86,  # HBISHOP
        6: 87,  # HROOK
    },
    WHITE: {
        0: 88,  # HPAWN
        1: 89,  # HLANCE
        2: 90,  # HKNIGHT
        3: 91,  # HSILVER
        4: 92,  # HGOLD
        5: 93,  # HBISHOP
        6: 94,  # HROOK
    },
}
