#!/usr/bin/env python3
"""Export a model-only checkpoint for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from posdiffusion.inference import export_inference_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Lightning training checkpoint")
    parser.add_argument("output", type=Path, help="Model-only output checkpoint")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_inference_checkpoint(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
