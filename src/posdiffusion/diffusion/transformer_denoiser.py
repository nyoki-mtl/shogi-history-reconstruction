"""Transformer-based denoiser for backward diffusion model."""

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import CombinedEmbedding


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear projections
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        # Transpose for attention: (batch_size, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads: (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Transpose back and reshape
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)

        # Final projection and dropout
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out  # type: ignore[no-any-return]


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (defaults to 4 * d_model)
            dropout: Dropout rate
            activation: Activation function name
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "gelu":
            self.activation: nn.Module = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function name
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out = self.attention(self.ln1(x), mask)
        x = x + attn_out

        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.ln2(x))
        x = x + ff_out

        return x


class ShogiTransformerDenoiser(nn.Module):
    """Main transformer denoiser for shogi position diffusion."""

    def __init__(
        self,
        vocab_size: int = 40,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int | None = None,
        max_positions: int = 96,
        max_timesteps: int = 100,
        max_moves: int = 400,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        use_game_progress: bool = False,
    ):
        """Initialize transformer denoiser.

        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_positions: Maximum number of positions
            max_timesteps: Maximum number of timesteps
            max_moves: Maximum number of moves in a game
            dropout: Dropout rate
            activation: Activation function name
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Embeddings
        self.embedding = CombinedEmbedding(
            d_model=d_model,
            vocab_size=vocab_size,
            max_positions=max_positions,
            max_timesteps=max_timesteps,
            max_moves=max_moves,
            dropout=dropout,
            use_game_progress=use_game_progress,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )

        # Output layer norm
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        noisy_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        absolute_move: torch.Tensor | None = None,
        game_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy_tokens: Noisy token indices of shape (batch_size, 96)
            timesteps: Timestep indices of shape (batch_size,)
            token_mask: Optional mask for valid tokens
            absolute_move: Optional absolute move numbers (batch_size,)
            game_phase: Optional game phase values [0.0-1.0] (batch_size,)

        Returns:
            Predicted logits of shape (batch_size, 96, vocab_size)
        """
        # Get embeddings
        x, _time_emb = self.embedding(noisy_tokens, timesteps, token_mask, absolute_move, game_phase)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, token_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits  # type: ignore[no-any-return]  # type: ignore[no-any-return]

    def predict_x0(
        self,
        noisy_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict clean tokens x0 from noisy tokens xt.

        Args:
            noisy_tokens: Noisy token indices of shape (batch_size, 96)
            timesteps: Timestep indices of shape (batch_size,)
            token_mask: Optional mask for valid tokens

        Returns:
            Predicted token probabilities of shape (batch_size, 96, vocab_size)
        """
        logits = self.forward(noisy_tokens, timesteps, token_mask)
        return F.softmax(logits, dim=-1)
