#!/usr/bin/env python3
"""Aggregate predecessor candidate coverage CSVs by ambiguity factors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoverageRow:
    game_id: str
    move_index: int
    move_bin: str
    candidate_count: int
    true_predecessor_covered: bool
    drop_candidate_count: int
    capture_candidate_count: int
    promotion_candidate_count: int
    nonpromotion_move_candidate_count: int
    successor_side_to_move_in_check: bool
    true_rank_rule_order: int | None
    true_kind: str | None
    true_is_drop: bool | None
    true_is_capture: bool | None
    true_is_promotion: bool | None


@dataclass(frozen=True)
class SummaryRow:
    group: str
    value: str
    num_positions: int
    true_predecessor_coverage: float
    candidate_count_mean: float
    candidate_count_median: float
    candidate_count_p10: float
    candidate_count_p90: float
    candidate_count_p99: float
    candidate_count_max: int
    drop_candidate_mean: float
    capture_candidate_mean: float
    promotion_candidate_mean: float
    nonpromotion_move_candidate_mean: float
    true_rank_median: float | None
    true_rank_p90: float | None


def parse_bool(raw: str) -> bool:
    if raw == "True":
        return True
    if raw == "False":
        return False
    raise ValueError(f"Expected boolean string, got {raw!r}")


def parse_optional_bool(raw: str) -> bool | None:
    return None if raw == "" else parse_bool(raw)


def parse_optional_int(raw: str) -> int | None:
    return None if raw == "" else int(raw)


def read_rows(paths: list[Path]) -> list[CoverageRow]:
    rows: list[CoverageRow] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    CoverageRow(
                        game_id=row["game_id"],
                        move_index=int(row["move_index"]),
                        move_bin=row["move_bin"],
                        candidate_count=int(row["candidate_count"]),
                        true_predecessor_covered=parse_bool(row["true_predecessor_covered"]),
                        drop_candidate_count=int(row["drop_candidate_count"]),
                        capture_candidate_count=int(row["capture_candidate_count"]),
                        promotion_candidate_count=int(row["promotion_candidate_count"]),
                        nonpromotion_move_candidate_count=int(row["nonpromotion_move_candidate_count"]),
                        successor_side_to_move_in_check=parse_bool(row["successor_side_to_move_in_check"]),
                        true_rank_rule_order=parse_optional_int(row["true_rank_rule_order"]),
                        true_kind=row["true_kind"] or None,
                        true_is_drop=parse_optional_bool(row["true_is_drop"]),
                        true_is_capture=parse_optional_bool(row["true_is_capture"]),
                        true_is_promotion=parse_optional_bool(row["true_is_promotion"]),
                    )
                )
    return rows


def quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def summarize_rows(group: str, value: str, rows: list[CoverageRow]) -> SummaryRow:
    counts = np.asarray([row.candidate_count for row in rows], dtype=np.float64)
    ranks = np.asarray(
        [row.true_rank_rule_order for row in rows if row.true_rank_rule_order is not None],
        dtype=np.float64,
    )
    covered = np.asarray([row.true_predecessor_covered for row in rows], dtype=np.bool_)
    return SummaryRow(
        group=group,
        value=value,
        num_positions=len(rows),
        true_predecessor_coverage=float(covered.mean()) if covered.size else 0.0,
        candidate_count_mean=float(counts.mean()) if counts.size else 0.0,
        candidate_count_median=float(np.median(counts)) if counts.size else 0.0,
        candidate_count_p10=float(np.quantile(counts, 0.1)) if counts.size else 0.0,
        candidate_count_p90=float(np.quantile(counts, 0.9)) if counts.size else 0.0,
        candidate_count_p99=float(np.quantile(counts, 0.99)) if counts.size else 0.0,
        candidate_count_max=int(counts.max()) if counts.size else 0,
        drop_candidate_mean=float(np.mean([row.drop_candidate_count for row in rows])) if rows else 0.0,
        capture_candidate_mean=float(np.mean([row.capture_candidate_count for row in rows])) if rows else 0.0,
        promotion_candidate_mean=float(np.mean([row.promotion_candidate_count for row in rows])) if rows else 0.0,
        nonpromotion_move_candidate_mean=(
            float(np.mean([row.nonpromotion_move_candidate_count for row in rows])) if rows else 0.0
        ),
        true_rank_median=quantile(ranks, 0.5),
        true_rank_p90=quantile(ranks, 0.9),
    )


def grouped_summaries(rows: list[CoverageRow]) -> list[SummaryRow]:
    group_specs: dict[str, Callable[[CoverageRow], str]] = {
        "all": lambda row: "all",
        "move_bin": lambda row: row.move_bin,
        "true_kind": lambda row: row.true_kind or "uncovered",
        "true_capture": lambda row: "capture" if row.true_is_capture else "non_capture",
        "true_promotion": lambda row: "promotion" if row.true_is_promotion else "non_promotion",
        "successor_check": lambda row: "check" if row.successor_side_to_move_in_check else "no_check",
    }
    summaries: list[SummaryRow] = []
    for group, key_fn in group_specs.items():
        buckets: dict[str, list[CoverageRow]] = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        for value in sorted(buckets):
            summaries.append(summarize_rows(group, value, buckets[value]))
    return summaries


def write_csv(path: Path, summaries: list[SummaryRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SummaryRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(asdict(row))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, summaries: list[SummaryRow]) -> None:
    lines = [
        "# Predecessor Candidate Ambiguity Aggregate",
        "",
        (
            "| Group | Value | Positions | Coverage | Candidate mean | Candidate p90 | "
            "Candidate p99 | Drop mean | Capture mean | Promotion mean | True rank median |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            f"{row.group} | "
            f"{row.value} | "
            f"{row.num_positions} | "
            f"{format_value(row.true_predecessor_coverage)} | "
            f"{format_value(row.candidate_count_mean)} | "
            f"{format_value(row.candidate_count_p90)} | "
            f"{format_value(row.candidate_count_p99)} | "
            f"{format_value(row.drop_candidate_mean)} | "
            f"{format_value(row.capture_candidate_mean)} | "
            f"{format_value(row.promotion_candidate_mean)} | "
            f"{format_value(row.true_rank_median)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate predecessor candidate coverage CSVs")
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.csv_paths)
    summaries = grouped_summaries(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "predecessor_candidate_ambiguity_aggregate_summary.json"
    csv_path = args.output_dir / "predecessor_candidate_ambiguity_aggregate_summary.csv"
    md_path = args.output_dir / "predecessor_candidate_ambiguity_aggregate_summary.md"
    json_path.write_text(
        json.dumps(
            {
                "num_input_files": len(args.csv_paths),
                "num_positions": len(rows),
                "summaries": [asdict(row) for row in summaries],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(csv_path, summaries)
    write_markdown(md_path, summaries)

    print(f"Rows aggregated: {len(rows)}")
    print(f"Summary JSON saved to: {json_path}")
    print(f"Summary CSV saved to: {csv_path}")
    print(f"Summary Markdown saved to: {md_path}")


if __name__ == "__main__":
    main()
