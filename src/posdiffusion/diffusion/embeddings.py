"""Embedding layers for backward diffusion model."""

import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Embedding for piece tokens.

    Handles 35 unique piece types:
    - 0: empty square
    - 1-14: black pieces (pawn, lance, knight, silver, gold, bishop, rook, king,
             promoted pawn, promoted lance, promoted knight, promoted silver,
             promoted bishop, promoted rook)
    - 15-28: white pieces (same order as black)
    - 29-34: captured pieces (pawn, lance, knight, silver, gold, bishop, rook) x2
    """

    def __init__(self, d_model: int = 768, vocab_size: int = 40):
        """Initialize token embedding.

        Args:
            d_model: Dimension of embeddings
            vocab_size: Size of vocabulary (40 for shogi pieces and practical hand pieces)
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Token indices of shape (batch_size, 96)

        Returns:
            Embeddings of shape (batch_size, 96, d_model)
        """
        return self.embedding(tokens)  # type: ignore[no-any-return]


class PositionEmbedding(nn.Module):
    """Position embeddings for 96 positions.

    Uses different embeddings for board squares (0-80) and captured pieces (81-94).
    """

    def __init__(self, d_model: int = 768, max_positions: int = 96):
        """Initialize position embedding.

        Args:
            d_model: Dimension of embeddings
            max_positions: Maximum number of positions (96)
        """
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions

        # Separate embeddings for board and captured pieces
        self.board_embedding = nn.Embedding(81, d_model)  # 9x9 board
        self.captured_embedding = nn.Embedding(14, d_model)  # 7 piece types x 2 colors

        # Initialize embeddings
        nn.init.normal_(self.board_embedding.weight, std=0.02)
        nn.init.normal_(self.captured_embedding.weight, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate position embeddings.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Position embeddings of shape (batch_size, 96, d_model)
        """
        # Board positions (0-80)
        board_positions = torch.arange(81, device=device)
        board_emb = self.board_embedding(board_positions)  # (81, d_model)

        # Captured positions (81-94)
        captured_positions = torch.arange(14, device=device)
        captured_emb = self.captured_embedding(captured_positions)  # (14, d_model)

        # Turn position (95)
        turn_position = torch.zeros(1, self.d_model, device=device)  # Placeholder for turn

        # Concatenate
        all_emb = torch.cat([board_emb, captured_emb, turn_position], dim=0)  # (96, d_model)

        # Expand for batch
        return all_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 96, d_model)


class TimeStepEmbedding(nn.Module):
    """Time step embedding using sinusoidal encoding + learnable projection.

    Similar to positional encoding in transformers but for diffusion timesteps.
    """

    def __init__(self, d_model: int = 768, max_timesteps: int = 100):
        """Initialize timestep embedding.

        Args:
            d_model: Dimension of embeddings
            max_timesteps: Maximum number of timesteps
        """
        super().__init__()
        self.d_model = d_model
        self.max_timesteps = max_timesteps

        # Sinusoidal embedding
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb_tensor = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb_tensor)

        # Learnable projection
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            timesteps: Timestep indices of shape (batch_size,)

        Returns:
            Time embeddings of shape (batch_size, d_model)
        """
        # Sinusoidal encoding
        emb = timesteps.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # Project through MLP
        return self.mlp(emb)  # type: ignore[no-any-return]


class GameProgressEmbedding(nn.Module):
    """Embeddings for game progress information.

    Handles:
    - absolute_move: Embedding for absolute move number in the game
    - game_phase: Linear projection for game phase (0.0 to 1.0)
    """

    def __init__(self, d_model: int = 768, max_moves: int = 400):
        """Initialize game progress embeddings.

        Args:
            d_model: Dimension of embeddings
            max_moves: Maximum number of moves in a game
        """
        super().__init__()
        self.d_model = d_model
        self.max_moves = max_moves

        # Absolute move embedding
        self.absolute_move_embed = nn.Embedding(max_moves, d_model)

        # Game phase embedding (continuous value 0.0-1.0)
        self.game_phase_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Initialize embeddings
        nn.init.normal_(self.absolute_move_embed.weight, std=0.02)

    def forward(
        self,
        absolute_move: torch.Tensor | None = None,
        game_phase: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Forward pass.

        Args:
            absolute_move: Absolute move numbers of shape (batch_size,)
            game_phase: Game phase values [0.0-1.0] of shape (batch_size,)

        Returns:
            Combined game progress embeddings of shape (batch_size, d_model)
            Returns None if both inputs are None
        """
        embeddings = []

        if absolute_move is not None:
            move_emb = self.absolute_move_embed(absolute_move)
            embeddings.append(move_emb)

        if game_phase is not None:
            # Reshape to (batch_size, 1) for linear layer
            phase_emb = self.game_phase_embed(game_phase.unsqueeze(-1))
            embeddings.append(phase_emb)

        if not embeddings:
            return None

        # Sum embeddings if both are present
        if len(embeddings) == 1:
            result: torch.Tensor = embeddings[0]
            return result
        else:
            result = embeddings[0] + embeddings[1]
            return result


class CombinedEmbedding(nn.Module):
    """Combines token, position, and timestep embeddings."""

    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 40,
        max_positions: int = 96,
        max_timesteps: int = 100,
        max_moves: int = 400,
        dropout: float = 0.1,
        use_game_progress: bool = False,
    ):
        """Initialize combined embedding.

        Args:
            d_model: Dimension of embeddings
            vocab_size: Size of token vocabulary
            max_positions: Maximum number of positions
            max_timesteps: Maximum number of timesteps
            max_moves: Maximum number of moves in a game
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.token_emb = TokenEmbedding(d_model, vocab_size)
        self.pos_emb = PositionEmbedding(d_model, max_positions)
        self.time_emb = TimeStepEmbedding(d_model, max_timesteps)
        self.game_progress_emb = GameProgressEmbedding(d_model, max_moves) if use_game_progress else None

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Scale factor for embeddings
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        absolute_move: torch.Tensor | None = None,
        game_phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            tokens: Token indices of shape (batch_size, 96)
            timesteps: Timestep indices of shape (batch_size,)
            token_mask: Optional mask for tokens (batch_size, 96)
            absolute_move: Optional absolute move numbers (batch_size,)
            game_phase: Optional game phase values [0.0-1.0] (batch_size,)

        Returns:
            Tuple of:
            - Combined embeddings of shape (batch_size, 96, d_model)
            - Time embeddings of shape (batch_size, d_model)
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Get embeddings
        token_emb = self.token_emb(tokens) * self.scale
        pos_emb = self.pos_emb(batch_size, device)
        time_emb = self.time_emb(timesteps)

        # Combine token and position embeddings
        combined = token_emb + pos_emb

        # Add time embedding to each position
        combined = combined + time_emb.unsqueeze(1)

        # Add game progress embeddings if enabled and provided
        if self.game_progress_emb is not None:
            game_progress_emb = self.game_progress_emb(absolute_move, game_phase)
            if game_progress_emb is not None:
                combined = combined + game_progress_emb.unsqueeze(1)

        # Apply layer norm and dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        # Apply mask if provided
        if token_mask is not None:
            combined = combined * token_mask.unsqueeze(-1)

        return combined, time_emb
