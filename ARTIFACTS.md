# Research Artifacts

## Paper

- **Proceedings page**: <https://ojs.aaai.org/index.php/AIIDE/article/view/43597>
- **PDF**: <https://ojs.aaai.org/index.php/AIIDE/article/view/43597/51155>
- **DOI**: <https://doi.org/10.1609/aiide.v22i1.43597>

## Model checkpoint (v1.0.0)

The trained predecessor model checkpoint is distributed via the [v1.0.0 GitHub release](https://github.com/nyoki-mtl/shogi-history-reconstruction/releases/tag/v1.0.0).

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| [`final_model.ckpt`](https://github.com/nyoki-mtl/shogi-history-reconstruction/releases/download/v1.0.0/final_model.ckpt) | 21,951,549 | `b07dab56ea90b5b80183fdc717ecff3c158f25c9bf2edc52025c4aebb370ae9a` |
| [`SHA256SUMS-v1.0.0.txt`](https://github.com/nyoki-mtl/shogi-history-reconstruction/releases/download/v1.0.0/SHA256SUMS-v1.0.0.txt) | - | release checksum manifest |

`final_model.ckpt` is an inference checkpoint containing the model configuration and weights (`state_dict`).

## External data and models

### Suisho 4T game records

Training and held-out evaluation use the public [Suisho 4T self-play KIF dataset](https://drive.google.com/file/d/10RuQMETwYclRggui0eb2CZGNVNVNxgdh/view). See [`README.md`](README.md#dataset-preparation) for shard creation and train/val/test split indices.

### DL-Suisho policy network (optional, for evaluation)

The forward-policy consistency diagnostics (`scripts/evaluation/evaluate_search_path_policy.py`) and forward-policy ranking control (`scripts/evaluation/evaluate_policy_forward_single_step_ranking.py`) use the public [DL-Suisho archive](https://drive.google.com/file/d/11e9-IhuUmZ68LaWE00m-_HFacsz8RgAX/view) (`DL水匠15b.zip`).

- Archive SHA-256: `6d675da7b0b460d7b70b50cd80475b90930a1876c27fbe0d2927545d9bc86944`
- Extract `DLSuisho15b/eval/model.onnx` and place it at `data/models/dl_suisho/model.onnx` (or pass `--alphazero_onnx` to the evaluation scripts).
- `model.onnx` SHA-256: `4c9e656f01b3ec692478965ee222a2116b73310368ed5aa499f1be1451484c0d`
