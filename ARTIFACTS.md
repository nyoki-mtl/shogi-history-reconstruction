# Research artifacts

Model weights are distributed with the tagged GitHub Release and identified by
SHA-256.

## Release v1.0.0

Canonical downloads:

<https://github.com/nyoki-mtl/shogi-history-reconstruction/releases/tag/v1.0.0>

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `final_model.ckpt` | 21,951,549 | `b07dab56ea90b5b80183fdc717ecff3c158f25c9bf2edc52025c4aebb370ae9a` |
| `SHA256SUMS-v1.0.0.txt` | - | release-level checksum manifest |

## Data availability

The repository does not redistribute game records. Training uses the public
[Suisho 4T self-play KIF dataset](https://drive.google.com/file/d/10RuQMETwYclRggui0eb2CZGNVNVNxgdh/view).

`final_model.ckpt` is an inference-only checkpoint containing the model
configuration and `state_dict`. Optimizer, logger, data-path, and trainer state
are intentionally excluded.
