# Legal predecessor selection

The rollback code is split across four modules:

- `candidates.py`: enumerate and verify legal predecessor candidates.
- `scoring.py`: score candidates from denoiser logits.
- `expansion.py`: cache candidate generation and construct ranked expansions.
- `search.py`: best-first and iterative widening beam rollback.

`rsshogi` provides the default candidate backend. Run
`scripts/setup_yaneuraou.sh` to build the YaneuraOu backend used in the paper's
multi-step evaluation.
