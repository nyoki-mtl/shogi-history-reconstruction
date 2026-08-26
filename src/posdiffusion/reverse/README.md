# Legal predecessor selection

This package implements the legal-projected rollback stack:

- `candidates.py`: enumerate and verify legal predecessor candidates.
- `scoring.py`: score candidates from denoiser logits.
- `expansion.py`: cache candidate generation and construct ranked expansions.
- `search.py`: best-first and iterative widening beam rollback.

Candidate generation is available through `rsshogi` and through the YaneuraOu
library built by `scripts/setup_yaneuraou.sh`.
