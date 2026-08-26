"""Base constants for shogi piece IDs.

This module defines constants that match the integer piece IDs used by the
project's 96-token representation.
"""

from typing import Final

# Define constants with the same values as the token representation.
# Empty square
NONE: Final[int] = 0

# Black pieces (1-14)
BPAWN: Final[int] = 1
BLANCE: Final[int] = 2
BKNIGHT: Final[int] = 3
BSILVER: Final[int] = 4
BBISHOP: Final[int] = 5
BROOK: Final[int] = 6
BGOLD: Final[int] = 7
BKING: Final[int] = 8
BPROM_PAWN: Final[int] = 9
BPROM_LANCE: Final[int] = 10
BPROM_KNIGHT: Final[int] = 11
BPROM_SILVER: Final[int] = 12
BPROM_BISHOP: Final[int] = 13
BPROM_ROOK: Final[int] = 14

# Unused (15-16)
NOTUSE: Final[int] = 15

# White pieces (17-30)
WPAWN: Final[int] = 17
WLANCE: Final[int] = 18
WKNIGHT: Final[int] = 19
WSILVER: Final[int] = 20
WBISHOP: Final[int] = 21
WROOK: Final[int] = 22
WGOLD: Final[int] = 23
WKING: Final[int] = 24
WPROM_PAWN: Final[int] = 25
WPROM_LANCE: Final[int] = 26
WPROM_KNIGHT: Final[int] = 27
WPROM_SILVER: Final[int] = 28
WPROM_BISHOP: Final[int] = 29
WPROM_ROOK: Final[int] = 30

# Colors
BLACK: Final[int] = 0
WHITE: Final[int] = 1

# Hand pieces (indices 0-6)
HPAWN: Final[int] = 0
HLANCE: Final[int] = 1
HKNIGHT: Final[int] = 2
HSILVER: Final[int] = 3
HGOLD: Final[int] = 4
HBISHOP: Final[int] = 5
HROOK: Final[int] = 6

# Alias for compatibility
EMPTY: Final[int] = NONE

# All piece IDs in order
ALL_PIECES: Final[tuple[int, ...]] = (
    NONE,
    BPAWN,
    BLANCE,
    BKNIGHT,
    BSILVER,
    BBISHOP,
    BROOK,
    BGOLD,
    BKING,
    BPROM_PAWN,
    BPROM_LANCE,
    BPROM_KNIGHT,
    BPROM_SILVER,
    BPROM_BISHOP,
    BPROM_ROOK,
    NOTUSE,
    NOTUSE,  # 15, 16 are unused
    WPAWN,
    WLANCE,
    WKNIGHT,
    WSILVER,
    WBISHOP,
    WROOK,
    WGOLD,
    WKING,
    WPROM_PAWN,
    WPROM_LANCE,
    WPROM_KNIGHT,
    WPROM_SILVER,
    WPROM_BISHOP,
    WPROM_ROOK,
)
