# YaneuraOu patch

`yaneuraou-posdiffusion.patch` adds the candidate-generation and C API changes
used in the paper. `scripts/setup_yaneuraou.sh` clones YaneuraOu into
`external/YaneuraOu/`, checks out revision
`014036fd2e5012693f2fe28e7a7cbd168a6e5b90`, applies the patch, and builds
the bindings.

YaneuraOu and this derivative patch are licensed under GNU GPL version 3; see
`YANEURAOU-GPL-3.0.txt`. The root MIT license covers the original code.
