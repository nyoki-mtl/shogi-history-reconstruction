# YaneuraOu patch

`yaneuraou-posdiffusion.patch` contains the candidate-generation and C API changes
used by the paper. It applies to public YaneuraOu revision
`014036fd2e5012693f2fe28e7a7cbd168a6e5b90`. The patched files are the exact
sources required by `scripts/build_yaneuraou_bindings.sh`.

YaneuraOu and this derivative patch are licensed under GNU GPL version 3. The
license text is included as `YANEURAOU-GPL-3.0.txt`. They are not covered by the
root MIT license. The setup script clones the upstream source into ignored local
path `external/YaneuraOu/`; no YaneuraOu source or binary is stored in this Git
repository.
