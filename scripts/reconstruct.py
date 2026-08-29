#!/usr/bin/env python3
"""Reconstruct a plausible legal history from a target SFEN position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rsshogi.core import Board

from posdiffusion.board.tokenizer import ShogiTokenizer
from posdiffusion.inference import load_model, path_to_usi
from posdiffusion.reverse.expansion import ExpansionProvider
from posdiffusion.reverse.search import iterative_widening_beam_rollback


def parse_widths(raw: str) -> tuple[int, ...]:
    widths = tuple(int(part) for part in raw.split(",") if part.strip())
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("Beam widths must be positive integers")
    return widths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-sfen", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidate-backend", choices=["rsshogi", "yaneuraou", "yaneuraou_hybrid"], default="rsshogi")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--beam-widths", default="1,3,10,30,100,300")
    parser.add_argument("--per-state-topk", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.max_steps <= 0 or args.budget <= 0 or args.per_state_topk <= 0:
        raise ValueError("Search limits must be positive")

    tokenizer = ShogiTokenizer(flip_to_black=False)
    initial_board = Board()
    target_board = Board()
    target_board.set_sfen(args.target_sfen)
    initial_tokens = tuple(tokenizer.encode(initial_board))
    target_tokens = tuple(tokenizer.encode(target_board))

    model = load_model(args.checkpoint, args.device)
    provider = ExpansionProvider(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        candidate_backend=args.candidate_backend,
        per_state_topk=args.per_state_topk,
        initial_tokens=initial_tokens,
    )
    result = iterative_widening_beam_rollback(
        start_tokens=target_tokens,
        initial_tokens=initial_tokens,
        expand_fn=provider.expand_fn("changed"),
        max_steps=args.max_steps,
        expansion_budget=args.budget,
        widths=parse_widths(args.beam_widths),
    )
    if not result.reached_initial or result.path is None:
        raise SystemExit(f"No history found: {result.stop_reason} after {result.expansions} expansions")

    history = path_to_usi(tokenizer, result.path)
    output = {
        "target_sfen": args.target_sfen,
        "generated_history_usi": history,
        "search": {
            "score_mode": "changed",
            "candidate_backend": args.candidate_backend,
            "beam_widths": list(parse_widths(args.beam_widths)),
            "expansion_budget": args.budget,
            "expansions_used": result.expansions,
            "steps_to_initial": result.steps_to_initial,
            "per_state_topk": args.per_state_topk,
        },
    }
    rendered = json.dumps(output, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
