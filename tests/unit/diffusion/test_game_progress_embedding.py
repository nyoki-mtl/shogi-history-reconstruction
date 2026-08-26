"""Test game progress embeddings functionality."""

import torch

from posdiffusion.diffusion.embeddings import (
    CombinedEmbedding,
    GameProgressEmbedding,
)
from posdiffusion.diffusion.transformer_denoiser import (
    ShogiTransformerDenoiser,
)


class TestGameProgressEmbedding:
    """Test GameProgressEmbedding class."""

    def test_initialization(self):
        """Test that GameProgressEmbedding initializes correctly."""
        d_model = 128
        max_moves = 300

        embedding = GameProgressEmbedding(d_model=d_model, max_moves=max_moves)

        assert embedding.d_model == d_model
        assert embedding.max_moves == max_moves
        assert embedding.absolute_move_embed.num_embeddings == max_moves
        assert embedding.absolute_move_embed.embedding_dim == d_model

    def test_forward_absolute_move_only(self):
        """Test forward pass with only absolute move."""
        embedding = GameProgressEmbedding(d_model=128, max_moves=300)

        batch_size = 4
        absolute_move = torch.randint(0, 100, (batch_size,))

        output = embedding(absolute_move=absolute_move, game_phase=None)

        assert output is not None
        assert output.shape == (batch_size, 128)

    def test_forward_game_phase_only(self):
        """Test forward pass with only game phase."""
        embedding = GameProgressEmbedding(d_model=128, max_moves=300)

        batch_size = 4
        game_phase = torch.rand(batch_size)

        output = embedding(absolute_move=None, game_phase=game_phase)

        assert output is not None
        assert output.shape == (batch_size, 128)

    def test_forward_both_inputs(self):
        """Test forward pass with both inputs."""
        embedding = GameProgressEmbedding(d_model=128, max_moves=300)

        batch_size = 4
        absolute_move = torch.randint(0, 100, (batch_size,))
        game_phase = torch.rand(batch_size)

        output = embedding(absolute_move=absolute_move, game_phase=game_phase)

        assert output is not None
        assert output.shape == (batch_size, 128)

    def test_forward_no_inputs(self):
        """Test forward pass with no inputs returns None."""
        embedding = GameProgressEmbedding(d_model=128, max_moves=300)

        output = embedding(absolute_move=None, game_phase=None)

        assert output is None


class TestCombinedEmbeddingWithGameProgress:
    """Test CombinedEmbedding with game progress features."""

    def test_initialization_with_max_moves(self):
        """Test that CombinedEmbedding initializes with max_moves parameter."""
        embedding = CombinedEmbedding(
            d_model=128,
            vocab_size=40,
            max_positions=96,
            max_timesteps=100,
            max_moves=400,
            use_game_progress=True,
        )

        assert hasattr(embedding, "game_progress_emb")
        assert embedding.game_progress_emb.max_moves == 400

    def test_forward_with_game_progress(self):
        """Test forward pass with game progress information."""
        embedding = CombinedEmbedding(
            d_model=128,
            vocab_size=40,
            max_positions=96,
            max_timesteps=100,
            max_moves=400,
            use_game_progress=True,
        )

        batch_size = 4
        tokens = torch.randint(0, 40, (batch_size, 96))
        timesteps = torch.randint(1, 20, (batch_size,))
        absolute_move = torch.randint(0, 100, (batch_size,))
        game_phase = torch.rand(batch_size)

        output, time_emb = embedding(
            tokens=tokens,
            timesteps=timesteps,
            absolute_move=absolute_move,
            game_phase=game_phase,
        )

        assert output.shape == (batch_size, 96, 128)
        assert time_emb.shape == (batch_size, 128)

    def test_forward_without_game_progress(self):
        """Test forward pass without game progress information."""
        embedding = CombinedEmbedding(
            d_model=128,
            vocab_size=40,
            max_positions=96,
            max_timesteps=100,
            max_moves=400,
        )

        batch_size = 4
        tokens = torch.randint(0, 40, (batch_size, 96))
        timesteps = torch.randint(1, 20, (batch_size,))

        output, time_emb = embedding(
            tokens=tokens,
            timesteps=timesteps,
            absolute_move=None,
            game_phase=None,
        )

        assert output.shape == (batch_size, 96, 128)
        assert time_emb.shape == (batch_size, 128)


class TestTransformerDenoiserWithGameProgress:
    """Test ShogiTransformerDenoiser with game progress features."""

    def test_initialization_with_max_moves(self):
        """Test that TransformerDenoiser initializes with max_moves parameter."""
        model = ShogiTransformerDenoiser(
            vocab_size=40,
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_moves=400,
            use_game_progress=True,
        )

        assert model.embedding.game_progress_emb.max_moves == 400

    def test_forward_with_game_progress(self):
        """Test forward pass with game progress information."""
        model = ShogiTransformerDenoiser(
            vocab_size=40,
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_moves=400,
            use_game_progress=True,
        )

        batch_size = 4
        tokens = torch.randint(0, 40, (batch_size, 96))
        timesteps = torch.randint(1, 20, (batch_size,))
        absolute_move = torch.randint(0, 100, (batch_size,))
        game_phase = torch.rand(batch_size)

        output = model(
            noisy_tokens=tokens,
            timesteps=timesteps,
            absolute_move=absolute_move,
            game_phase=game_phase,
        )

        assert output.shape == (batch_size, 96, 40)

    def test_forward_without_game_progress(self):
        """Test forward pass without game progress information."""
        model = ShogiTransformerDenoiser(
            vocab_size=40,
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_moves=400,
        )

        batch_size = 4
        tokens = torch.randint(0, 40, (batch_size, 96))
        timesteps = torch.randint(1, 20, (batch_size,))

        output = model(
            noisy_tokens=tokens,
            timesteps=timesteps,
            absolute_move=None,
            game_phase=None,
        )

        assert output.shape == (batch_size, 96, 40)

    def test_backward_compatibility(self):
        """Test that the model works without specifying max_moves."""
        # This should use the default max_moves value
        model = ShogiTransformerDenoiser(
            vocab_size=40,
            d_model=128,
            n_heads=4,
            n_layers=2,
        )

        batch_size = 4
        tokens = torch.randint(0, 40, (batch_size, 96))
        timesteps = torch.randint(1, 20, (batch_size,))

        output = model(noisy_tokens=tokens, timesteps=timesteps)

        assert output.shape == (batch_size, 96, 40)
