"""Model loading and path conversion for history reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.diffusion.transformer_denoiser import ShogiTransformerDenoiser
from posdiffusion.reverse.search import Tokens

INFERENCE_CHECKPOINT_FORMAT = "shogi-history-reconstruction.inference.v1"


def _model_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    if checkpoint.get("format") == INFERENCE_CHECKPOINT_FORMAT:
        model_config = checkpoint.get("model_config")
    else:
        hyper_parameters = checkpoint.get("hyper_parameters")
        model_config = (
            hyper_parameters.get("model", hyper_parameters) if isinstance(hyper_parameters, Mapping) else None
        )
    if not isinstance(model_config, Mapping):
        raise ValueError("Checkpoint does not contain a model configuration")
    return model_config


def _model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a state_dict")
    if checkpoint.get("format") == INFERENCE_CHECKPOINT_FORMAT:
        return dict(state_dict)
    return {key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")}


def load_model(checkpoint_path: str | Path, device: str) -> ShogiTransformerDenoiser:
    """Load either the released inference checkpoint or a Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    model_config = _model_config(checkpoint)
    state_dict = _model_state(checkpoint)
    use_game_progress = bool(model_config.get("use_game_progress", False)) or any(
        key.startswith("embedding.game_progress_emb.") for key in state_dict
    )
    model = ShogiTransformerDenoiser(
        vocab_size=int(model_config.get("vocab_size", 40)),
        d_model=int(model_config.get("d_model", 768)),
        n_heads=int(model_config.get("n_heads", 12)),
        n_layers=int(model_config.get("n_layers", 12)),
        d_ff=model_config.get("d_ff"),
        max_positions=int(model_config.get("max_positions", 96)),
        max_timesteps=int(model_config.get("num_timesteps", 100)),
        max_moves=int(model_config.get("max_moves", 400)),
        dropout=float(model_config.get("dropout", 0.1)),
        use_game_progress=use_game_progress,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def export_inference_checkpoint(source_path: str | Path, output_path: str | Path) -> None:
    """Strip training state from a Lightning checkpoint."""
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    model_config = dict(_model_config(checkpoint))
    state_dict = _model_state(checkpoint)
    exported = {
        "format": INFERENCE_CHECKPOINT_FORMAT,
        "model_config": model_config,
        "state_dict": state_dict,
    }
    torch.save(exported, output_path)


def _sfen_core(tokens: Tokens, tokenizer: ShogiTokenizer) -> str:
    return " ".join(tokenizer.decode(list(tokens)).to_sfen().split()[:3])


def find_forward_move(tokenizer: ShogiTokenizer, predecessor: Tokens, successor: Tokens) -> str:
    """Return the legal USI move connecting two adjacent path states."""
    board = tokenizer.decode(list(predecessor))
    target = _sfen_core(successor, tokenizer)
    for move in board.legal_moves():
        board.push_move(move)
        matches = " ".join(board.to_sfen().split()[:3]) == target
        board.pop()
        if matches:
            return move.to_usi()
    raise ValueError("No legal move connects the adjacent path states")


def path_to_usi(tokenizer: ShogiTokenizer, rollback_path: list[Tokens]) -> list[str]:
    """Convert a rollback path ``[target, ..., initial]`` to forward USI moves."""
    forward_states = list(reversed(rollback_path))
    return [
        find_forward_move(tokenizer, forward_states[index], forward_states[index + 1])
        for index in range(len(forward_states) - 1)
    ]
