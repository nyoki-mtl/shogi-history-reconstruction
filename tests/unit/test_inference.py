from __future__ import annotations

import torch

from posdiffusion.diffusion.transformer_denoiser import ShogiTransformerDenoiser
from posdiffusion.inference import INFERENCE_CHECKPOINT_FORMAT, export_inference_checkpoint, load_model


def test_exported_checkpoint_contains_only_inference_state(tmp_path) -> None:
    config = {
        "vocab_size": 40,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "d_ff": 32,
        "dropout": 0.0,
        "num_timesteps": 10,
    }
    model = ShogiTransformerDenoiser(
        vocab_size=40,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        max_timesteps=10,
    )
    source = tmp_path / "training.ckpt"
    output = tmp_path / "inference.ckpt"
    torch.save(
        {
            "hyper_parameters": {"model": config, "data": {"train_shard_pattern": "private/path"}},
            "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
            "optimizer_states": [{"not": "published"}],
        },
        source,
    )

    export_inference_checkpoint(source, output)

    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"format", "model_config", "state_dict"}
    assert checkpoint["format"] == INFERENCE_CHECKPOINT_FORMAT
    assert "data" not in checkpoint["model_config"]
    loaded = load_model(output, "cpu")
    inputs = torch.zeros((1, 96), dtype=torch.long)
    timesteps = torch.ones((1,), dtype=torch.long)
    torch.testing.assert_close(loaded(inputs, timesteps), model(inputs, timesteps))
