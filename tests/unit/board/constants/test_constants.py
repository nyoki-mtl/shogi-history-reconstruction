"""Test unified constants module."""

from posdiffusion.board.constants import (
    BBISHOP,
    BGOLD,
    BKING,
    BKNIGHT,
    BLACK,
    BLANCE,
    BPAWN,
    BROOK,
    BSILVER,
    EMPTY,
    NONE,
    WBISHOP,
    WGOLD,
    WHITE,
    WKING,
    WKNIGHT,
    WLANCE,
    WPAWN,
    WROOK,
    WSILVER,
)


class TestBasicConstants:
    """Test basic constant values used by the token representation."""

    def test_empty_constants(self):
        """Test empty square constants."""
        assert NONE == 0
        assert EMPTY == 0
        assert NONE == EMPTY

    def test_black_pieces(self):
        """Test black piece constants."""
        assert BPAWN == 1
        assert BLANCE == 2
        assert BKNIGHT == 3
        assert BSILVER == 4
        assert BBISHOP == 5
        assert BROOK == 6
        assert BGOLD == 7
        assert BKING == 8

    def test_white_pieces(self):
        """Test white piece constants."""
        assert WPAWN == 17
        assert WLANCE == 18
        assert WKNIGHT == 19
        assert WSILVER == 20
        assert WBISHOP == 21
        assert WROOK == 22
        assert WGOLD == 23
        assert WKING == 24

    def test_colors(self):
        """Test color constants."""
        assert BLACK == 0
        assert WHITE == 1
