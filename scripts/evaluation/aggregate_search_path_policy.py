#!/usr/bin/env python3
"""Aggregate sharded search path policy CSVs into one summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _floats(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key)]


def aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_setting: dict[str, dict[str, Any]] = {}
    for start_move in sorted({int(row["start_move"]) for row in rows}):
        selected = [row for row in rows if int(row["start_move"]) == start_move]
        reached = [row for row in selected if row["reached_initial"] == "True"]
        model_means = _floats(reached, "model_path_policy_mean")
        true_means_all = _floats(selected, "true_path_policy_mean")
        true_means_success = _floats(reached, "true_path_policy_mean")
        exact_steps = _floats(reached, "model_exact_step_count")
        steps = _floats(reached, "steps_to_initial")
        by_setting[f"move{start_move}"] = {
            "num_records": len(selected),
            "reached_rate": len(reached) / len(selected),
            "model_path_policy_mean_avg": float(np.mean(model_means)) if model_means else None,
            "model_path_policy_mean_median": float(median(model_means)) if model_means else None,
            "true_path_policy_mean_avg_all": float(np.mean(true_means_all)) if true_means_all else None,
            "true_path_policy_mean_avg_success_games": float(np.mean(true_means_success))
            if true_means_success
            else None,
            "model_minus_true_on_success": float(np.mean(model_means) - np.mean(true_means_success))
            if model_means and true_means_success
            else None,
            "model_exact_step_rate": float(np.sum(exact_steps) / np.sum(steps)) if steps else None,
        }
    return {"num_rows": len(rows), "by_setting": by_setting}


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        (
            "| Start move | Records | Reached | Model policy mean "
            "| True policy mean (same games) | Model - True | Exact-step rate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in summary["by_setting"].items():

        def fmt(name: str, value: dict[str, Any] = value) -> str:
            item = value[name]
            return f"{item:.4f}" if item is not None else "-"

        lines.append(
            f"| {key} | {value['num_records']} | {value['reached_rate']:.3f} "
            f"| {fmt('model_path_policy_mean_avg')} | {fmt('true_path_policy_mean_avg_success_games')} "
            f"| {fmt('model_minus_true_on_success')} | {fmt('model_exact_step_rate')} |"
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
    (args.output_dir / "search_path_policy_aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "search_path_policy_aggregate_summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    print(f"Aggregated {summary['num_rows']} rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
