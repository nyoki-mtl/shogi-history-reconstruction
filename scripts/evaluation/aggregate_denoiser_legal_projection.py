#!/usr/bin/env python3
"""Aggregate denoiser legal-projection CSVs by move bins."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectionRow:
    game_id: str
    move_index: int
    mode: str
    candidate_count: int
    true_predecessor_covered: bool
    true_rank: int | None
    top1_exact: bool
    top3_exact: bool
    top5_exact: bool


@dataclass(frozen=True)
class BaselineRow:
    game_id: str
    move_index: int
    baseline: str
    candidate_count: int
    true_predecessor_covered: bool
    top1_exact: bool
    selected_is_reverse_legal: bool


def parse_bool(raw: str) -> bool:
    if raw == "True":
        return True
    if raw == "False":
        return False
    raise ValueError(f"Expected boolean string, got {raw!r}")


def parse_optional_int(raw: str) -> int | None:
    return None if raw == "" else int(raw)


def read_projection_rows(paths: list[Path]) -> list[ProjectionRow]:
    rows: list[ProjectionRow] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    ProjectionRow(
                        game_id=row["game_id"],
                        move_index=int(row["move_index"]),
                        mode=row["mode"],
                        candidate_count=int(row["candidate_count"]),
                        true_predecessor_covered=parse_bool(row["true_predecessor_covered"]),
                        true_rank=parse_optional_int(row["true_rank"]),
                        top1_exact=parse_bool(row["top1_exact"]),
                        top3_exact=parse_bool(row["top3_exact"]),
                        top5_exact=parse_bool(row["top5_exact"]),
                    )
                )
    return rows


def read_baseline_rows(paths: list[Path]) -> list[BaselineRow]:
    rows: list[BaselineRow] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    BaselineRow(
                        game_id=row["game_id"],
                        move_index=int(row["move_index"]),
                        baseline=row["baseline"],
                        candidate_count=int(row["candidate_count"]),
                        true_predecessor_covered=parse_bool(row["true_predecessor_covered"]),
                        top1_exact=parse_bool(row["top1_exact"]),
                        selected_is_reverse_legal=parse_bool(row["selected_is_reverse_legal"]),
                    )
                )
    return rows


def bin_label(move_index: int, bin_size: int) -> str:
    if bin_size <= 0:
        return str(move_index)
    start = ((move_index - 1) // bin_size) * bin_size + 1
    end = start + bin_size - 1
    return f"{start:03d}-{end:03d}"


def quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def summarize_projection(rows: list[ProjectionRow], bin_size: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ProjectionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.mode, bin_label(row.move_index, bin_size))].append(row)

    summaries: list[dict[str, Any]] = []
    for (mode, move_bin), selected in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        covered = [row for row in selected if row.true_predecessor_covered]
        counts = np.asarray([row.candidate_count for row in selected], dtype=np.float64)
        ranks = np.asarray([row.true_rank for row in covered if row.true_rank is not None], dtype=np.float64)
        summaries.append(
            {
                "kind": "projection",
                "name": mode,
                "move_bin": move_bin,
                "records": len(selected),
                "coverage": len(covered) / len(selected) if selected else 0.0,
                "top1": sum(row.top1_exact for row in selected) / len(selected) if selected else 0.0,
                "top3": sum(row.top3_exact for row in selected) / len(selected) if selected else 0.0,
                "top5": sum(row.top5_exact for row in selected) / len(selected) if selected else 0.0,
                "covered_top1": sum(row.top1_exact for row in covered) / len(covered) if covered else 0.0,
                "covered_top3": sum(row.top3_exact for row in covered) / len(covered) if covered else 0.0,
                "covered_top5": sum(row.top5_exact for row in covered) / len(covered) if covered else 0.0,
                "candidate_count_mean": float(counts.mean()) if counts.size else 0.0,
                "candidate_count_median": quantile(counts, 0.5),
                "candidate_count_p90": quantile(counts, 0.9),
                "candidate_count_p99": quantile(counts, 0.99),
                "true_rank_median": quantile(ranks, 0.5),
                "true_rank_p90": quantile(ranks, 0.9),
            }
        )
    return summaries


def summarize_baselines(rows: list[BaselineRow], bin_size: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[BaselineRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.baseline, bin_label(row.move_index, bin_size))].append(row)

    summaries: list[dict[str, Any]] = []
    for (baseline, move_bin), selected in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        covered = [row for row in selected if row.true_predecessor_covered]
        legal = [row for row in selected if row.selected_is_reverse_legal]
        counts = np.asarray([row.candidate_count for row in selected], dtype=np.float64)
        summaries.append(
            {
                "kind": "baseline",
                "name": baseline,
                "move_bin": move_bin,
                "records": len(selected),
                "coverage": len(covered) / len(selected) if selected else 0.0,
                "top1": sum(row.top1_exact for row in selected) / len(selected) if selected else 0.0,
                "top3": None,
                "top5": None,
                "covered_top1": sum(row.top1_exact for row in covered) / len(covered) if covered else 0.0,
                "covered_top3": None,
                "covered_top5": None,
                "candidate_count_mean": float(counts.mean()) if counts.size else 0.0,
                "candidate_count_median": quantile(counts, 0.5),
                "candidate_count_p90": quantile(counts, 0.9),
                "candidate_count_p99": quantile(counts, 0.99),
                "true_rank_median": None,
                "true_rank_p90": None,
                "selected_reverse_legal_rate": len(legal) / len(selected) if selected else 0.0,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kind",
        "name",
        "move_bin",
        "records",
        "coverage",
        "top1",
        "top3",
        "top5",
        "covered_top1",
        "covered_top3",
        "covered_top5",
        "candidate_count_mean",
        "candidate_count_median",
        "candidate_count_p90",
        "candidate_count_p99",
        "true_rank_median",
        "true_rank_p90",
        "selected_reverse_legal_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Denoiser Legal Projection Aggregate",
        "",
        (
            "| Kind | Name | Move bin | Records | Coverage | Top-1 | Top-3 | Top-5 | "
            "Candidate mean | Candidate p90 | Rank median | Rank p90 |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['kind']} | "
            f"{row['name']} | "
            f"{row['move_bin']} | "
            f"{row['records']} | "
            f"{format_value(row['coverage'])} | "
            f"{format_value(row['top1'])} | "
            f"{format_value(row['top3'])} | "
            f"{format_value(row['top5'])} | "
            f"{format_value(row['candidate_count_mean'])} | "
            f"{format_value(row['candidate_count_p90'])} | "
            f"{format_value(row['true_rank_median'])} | "
            f"{format_value(row['true_rank_p90'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate denoiser legal projection CSVs")
    parser.add_argument("--projection_csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline_csvs", nargs="*", type=Path, default=[])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--move_bin_size", type=int, default=10, help="0 means exact move index")
    args = parser.parse_args()
    if args.move_bin_size < 0:
        raise ValueError("--move_bin_size must be non-negative")

    projection_rows = read_projection_rows(args.projection_csvs)
    baseline_rows = read_baseline_rows(args.baseline_csvs)
    summaries = summarize_projection(projection_rows, args.move_bin_size) + summarize_baselines(
        baseline_rows,
        args.move_bin_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "denoiser_legal_projection_aggregate_summary.json"
    csv_path = args.output_dir / "denoiser_legal_projection_aggregate_summary.csv"
    md_path = args.output_dir / "denoiser_legal_projection_aggregate_summary.md"
    json_path.write_text(
        json.dumps(
            {
                "num_projection_rows": len(projection_rows),
                "num_baseline_rows": len(baseline_rows),
                "move_bin_size": args.move_bin_size,
                "by_kind_name_move_bin": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(csv_path, summaries)
    write_markdown(md_path, summaries)

    print(f"Projection rows aggregated: {len(projection_rows)}")
    print(f"Baseline rows aggregated: {len(baseline_rows)}")
    print(f"Summary JSON saved to: {json_path}")
    print(f"Summary CSV saved to: {csv_path}")
    print(f"Summary Markdown saved to: {md_path}")


if __name__ == "__main__":
    main()
