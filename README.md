# Shogi History Reconstruction

Code and model weights for the AIIDE 2026 paper:

**Reconstructing Plausible Game Histories via Legal-Projected Reverse Generation: A Shogi Case Study**<br>
Hiroki Taniai and Hidetsugu Irie (The University of Tokyo)<br>
*Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE)*, Vol. 22, No. 1, pp. 394-402, 2026.

[[Paper (AAAI OJS)]](https://ojs.aaai.org/index.php/AIIDE/article/view/43597) | [[PDF]](https://ojs.aaai.org/index.php/AIIDE/article/view/43597/51155) | [[DOI: 10.1609/aiide.v22i1.43597]](https://doi.org/10.1609/aiide.v22i1.43597) | [[Model Checkpoint (v1.0.0)]](https://github.com/nyoki-mtl/shogi-history-reconstruction/releases/tag/v1.0.0)

## Overview

Given a target shogi position without its move log, this codebase reconstructs a legal game history from the standard initial setup (`s_0`) to the target (`s_T`).

Direct neural prediction of the previous board state often fails during multi-step rollback because a single illegal transition invalidates the entire history (on 96,023 held-out test states, token-wise argmax is statically valid on 85.5% of states but legally reachable in one move on only 76.5%). Instead, we combine exact reverse move generation with learned ranking:

1. **Enumerate legal predecessors**: generate candidate states `c` that can reach the current state `s_{t+1}` by a legal shogi move (121 candidates per step on average), and verify each candidate by forward move replay.
2. **Score changed tokens**: run a 5.5M-parameter bidirectional Transformer over the 96-token state representation (`81` board squares + `7` Black hand counts + `7` White hand counts + `1` side-to-move) and rank candidates by the sum of log-probabilities over the tokens that differ from `s_{t+1}`. All candidates at a given step reuse a single forward pass.
3. **Search backward**: apply iterative widening beam search (widths `1, 3, 10, 30, 100`, plus `300` for start depths `T >= 80`) with admissible lower-bound pruning on the remaining steps to `s_0`.

![Legal-projected reverse generation pipeline](docs/images/method-overview.png)

## Worked example

<img src="docs/images/worked-example-target.png" alt="Target shogi position" width="480">

Target SFEN (from a 100-move test game, move-count field omitted):

```text
ln5nl/4rgg1k/p2p4s/1s3PP1p/2PPP1RP1/P1GB5/1PN2SN1P/1SK1G4/L7+b b 2Pl4p
```

From this position, the width-1 pass reaches the standard initial position in 80 steps (80 state expansions, no backtracking). Replaying the 80 USI moves below from the initial setup reproduces the target board, hands, and side to move:

```text
2g2f 8c8d 7g7f 8d8e 8h7g 3c3d 7i8h 4a3b 3i4h 7a6b 6i7h 3a4b 5i6i 7c7d 9g9f 4b3c 3g3f 2b3a 4i5h 5a4a 5g5f 6a5b 2i3g 5c5d 4g4f 1c1d 2f2e 4c4d 6g6f 8e8f 7g8f 3a8f 8g8f 8b8f P*8g 8f8b 4h4g 6b7c 7h6g 7c6d 6i7h 5d5e 5f5e 6d5e P*5f 5e6d 6f6e 6d7c 8i7g 7c8d 5f5e 7d7e B*6f 7e7f 6g7f 4a3a P*7e 5b4b 3f3e 8b5b 3e3d 3c2b 2e2d 2c2d 2h2d 2b2c 2d2f 3a2b 2f3f B*2h 4f4e 2h1i+ P*2d 2c2d P*2e 2d1c 4e4d P*3e 3f3e 2b1b
```

The full output and search metadata are in [`docs/worked-example.json`](docs/worked-example.json).

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
```

Download [`final_model.ckpt`](https://github.com/nyoki-mtl/shogi-history-reconstruction/releases/download/v1.0.0/final_model.ckpt) from the [v1.0.0 release](https://github.com/nyoki-mtl/shogi-history-reconstruction/releases/tag/v1.0.0) (checksums in [`ARTIFACTS.md`](ARTIFACTS.md)), place it in the repository root, and run:

```bash
uv run python scripts/reconstruct.py \
  --checkpoint final_model.ckpt \
  --target-sfen 'ln5nl/4rgg1k/p2p4s/1s3PP1p/2PPP1RP1/P1GB5/1PN2SN1P/1SK1G4/L7+b b 2Pl4p'
```

By default, `scripts/reconstruct.py` uses [`rsshogi`](https://pypi.org/project/rsshogi/) for legal predecessor enumeration, `changed`-token scoring, beam widths `1,3,10,30,100,300`, `per_state_topk=10`, and an expansion budget of `20000`.

For the batch multi-step rollback benchmarks reported in the paper, build the patched YaneuraOu C++ candidate-generation backend (requires `git` and `clang++`):

```bash
bash scripts/setup_yaneuraou.sh
```

Then pass `--candidate-backend yaneuraou` to `scripts/reconstruct.py` or the evaluation scripts.

## Training and evaluation

### Dataset preparation

Training and held-out evaluation use the public [Suisho 4T self-play KIF dataset](https://drive.google.com/file/d/10RuQMETwYclRggui0eb2CZGNVNVNxgdh/view). Extract the `.kif` files under `data/2mkif/` and build WebDataset shards truncated to the first 100 moves:

```bash
uv run python scripts/data/create_webdataset.py \
  --kif-dir data/2mkif \
  --output-dir data/2mkif_shards \
  --shard-size 10000 \
  --max-moves 100
```

The paper split uses shards `00000..00105` for training, `00106..00107` for validation, and `00108..00109` for held-out evaluation.

### Model training

```bash
uv run python scripts/train.py \
  --config configs/experiments/constrained.yaml \
  'data.train_shard_pattern=data/2mkif_shards/shard_{00000..00105}.tar' \
  'data.val_shard_pattern=data/2mkif_shards/shard_{00106..00107}.tar' \
  'data.test_shard_pattern=data/2mkif_shards/shard_{00108..00109}.tar'
```

`configs/experiments/constrained.yaml` trains the soft-constrained predecessor model (token cross-entropy plus soft piece-count and step-size losses). `configs/experiments/baseline.yaml` trains the unconstrained cross-entropy baseline.

### Evaluation scripts

Scripts for the paper's tables and figures are in `scripts/evaluation/`:

- `argmax_eval_full.py`: single-step direct argmax evaluation (`validator-legal` and `reverse-legal`).
- `evaluate_denoiser_legal_projection.py` / `aggregate_denoiser_legal_projection.py`: single-step legal predecessor ranking (Table 2, Figure 2).
- `evaluate_projection_search_rollback.py` / `aggregate_projection_search_results.py`: multi-step iterative widening rollback across budgets and start depths (Table 3, Figure 3).
- `evaluate_search_path_policy.py` / `aggregate_search_path_policy.py`: forward-policy consistency diagnostics on reached paths (Table 4).
- `evaluate_policy_forward_single_step_ranking.py`: forward-policy single-step ranking control.

See [`ARTIFACTS.md`](ARTIFACTS.md) for the optional DL-Suisho policy network (`model.onnx`) used by the forward-policy scripts.

## Citation

```bibtex
@article{Taniai_Irie_2026,
  title={Reconstructing Plausible Game Histories via Legal-Projected Reverse Generation: A Shogi Case Study},
  author={Taniai, Hiroki and Irie, Hidetsugu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  volume={22},
  number={1},
  pages={394--402},
  year={2026},
  month={Nov.},
  doi={10.1609/aiide.v22i1.43597},
  url={https://ojs.aaai.org/index.php/AIIDE/article/view/43597}
}
```

## License

Original code in this repository is licensed under the [MIT License](LICENSE). YaneuraOu and the patch in [`third_party/`](third_party/README.md) are licensed under GNU GPL v3.0; other dependencies retain their own terms.
