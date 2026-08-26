#!/usr/bin/env python3
"""Aggregate sharded projection search rollback CSVs into one summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    keys = sorted(
        {
            (
                int(row["start_move"]),
                row["mode"],
                int(row["expansion_budget"]),
                float(row["step_cost"]),
                float(row.get("step_bonus") or 0.0),
            )
            for row in rows
        }
    )
    by_setting: dict[str, dict[str, Any]] = {}
    for start_move, mode, budget, step_cost, step_bonus in keys:
        selected = [
            row
            for row in rows
            if int(row["start_move"]) == start_move
            and row["mode"] == mode
            and int(row["expansion_budget"]) == budget
            and float(row["step_cost"]) == step_cost
            and float(row.get("step_bonus") or 0.0) == step_bonus
        ]
        reached = [row for row in selected if row["reached_initial"] == "True"]
        by_setting[f"move{start_move}/{mode}/budget{budget}/cost{step_cost:g}/bonus{step_bonus:g}"] = {
            "num_records": len(selected),
            "reached_rate": len(reached) / len(selected),
            "exact_path_rate": sum(row["exact_path"] == "True" for row in selected) / len(selected),
            "avg_expansions": float(np.mean([int(row["expansions_used"]) for row in selected])),
            "avg_expansions_success": float(np.mean([int(row["expansions_used"]) for row in reached]))
            if reached
            else None,
            "avg_steps_to_initial": float(np.mean([int(row["steps_to_initial"]) for row in reached]))
            if reached
            else None,
            "avg_true_state_overlap_success": float(
                np.mean([float(row["true_state_overlap"]) for row in reached if row.get("true_state_overlap")])
            )
            if reached
            else None,
            "avg_elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in selected])),
            "stop_reasons": {
                reason: sum(1 for row in selected if row["stop_reason"] == reason)
                for reason in sorted({row["stop_reason"] for row in selected})
            },
        }
    return {"num_rows": len(rows), "by_setting": by_setting}


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        (
            "| Setting | Records | Reached | Exact path | Overlap (succ) "
            "| Avg expansions | Avg exp (success) | Avg steps | Stop reasons |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, value in summary["by_setting"].items():
        exp_success = f"{value['avg_expansions_success']:.1f}" if value["avg_expansions_success"] is not None else "-"
        steps = f"{value['avg_steps_to_initial']:.1f}" if value["avg_steps_to_initial"] is not None else "-"
        overlap = (
            f"{value['avg_true_state_overlap_success']:.3f}"
            if value["avg_true_state_overlap_success"] is not None
            else "-"
        )
        lines.append(
            f"| {key} | {value['num_records']} | {value['reached_rate']:.3f} | {value['exact_path_rate']:.3f} "
            f"| {overlap} | {value['avg_expansions']:.1f} | {exp_success} | {steps} "
            f"| {json.dumps(value['stop_reasons'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.csv_paths)
    if not rows:
        raise ValueError("No rows found in the given CSV files")
    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "projection_search_rollback_aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "projection_search_rollback_aggregate_summary.md").write_text(
        to_markdown(summary), encoding="utf-8"
    )
    print(to_markdown(summary))
    print(f"Aggregated {summary['num_rows']} rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
