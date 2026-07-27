"""Aggregate five frozen paired iDDPM endpoint seeds."""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARTIFACT = ROOT / ".openresearch" / "artifacts" / HERE.name
INPUT = HERE / "inputs" / "endpoints.json"


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(differences: list[float]) -> dict:
    means = [
        sum(differences[index] for index in sample) / len(differences)
        for sample in itertools.product(range(len(differences)), repeat=len(differences))
    ]
    mean = sum(differences) / len(differences)
    sample_variance = sum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    interval = [quantile(means, 0.025), quantile(means, 0.975)]
    return {
        "paired_differences": differences,
        "mean": mean,
        "sample_sd": math.sqrt(sample_variance),
        "exhaustive_bootstrap_resamples": len(means),
        "percentile_95_interval": interval,
        "all_observed_positive": all(value > 0 for value in differences),
        "accepted": all(value > 0 for value in differences) and interval[0] > 0,
    }


def calculate(rows: list[dict], reverse: bool = False) -> dict:
    grouped: dict[int, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["seed_index"], {})[row["endpoint"]] = row
    result = {}
    for metric in ("sw2", "msw2"):
        differences = []
        for seed_index in range(5):
            pair = grouped[seed_index]
            first = pair["smallest"] if reverse else pair["largest"]
            second = pair["largest"] if reverse else pair["smallest"]
            differences.append(first[metric] - second[metric])
        result[metric] = summarize(differences)
    result["accepted"] = all(result[metric]["accepted"] for metric in ("sw2", "msw2"))
    return result


def main() -> None:
    started = time.perf_counter()
    source = json.loads(INPUT.read_text())
    rows = source["endpoints"]
    grouped: dict[int, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["seed_index"], {})[row["endpoint"]] = row
    input_checks = {
        "ten_endpoints": len(rows) == 10,
        "five_seed_indices": sorted(grouped) == list(range(5)),
        "complete_pairs": all(set(grouped[index]) == {"largest", "smallest"} for index in grouped),
        "paired_seed_ids_match": all(
            grouped[index]["largest"]["paired_seed"]
            == grouped[index]["smallest"]["paired_seed"]
            for index in grouped
        ),
        "unique_run_ids": len({row["run_id"] for row in rows}) == 10,
        "metrics_finite": all(
            math.isfinite(row[metric]) for row in rows for metric in ("sw2", "msw2")
        ),
        "endpoint_checkers_passed": all(
            row["exact_setup_passed"] and row["independent_checker_passed"]
            for row in rows
        ),
        "historical_geometry_disclosed": source["geometry"]["network_count"] == 16_000,
    }
    primary = calculate(rows)
    reversed_control = calculate(rows, reverse=True)
    control = {
        "name": "reverse endpoint labels",
        "expected_to_fail": True,
        "accepted_under_positive_effect_rule": reversed_control["accepted"],
        "rejected": not reversed_control["accepted"],
        "sw2_mean": reversed_control["sw2"]["mean"],
        "msw2_mean": reversed_control["msw2"]["mean"],
    }
    passed = all(input_checks.values()) and primary["accepted"] and control["rejected"]
    result = {
        "claims": [1, 4],
        "scientific_status": "HISTORICAL_DOWNSCALED_CORROBORATION",
        "claim_verdict": "BLOCKED_PENDING_PAPER_SCALE_GENERATION_AND_RANK_ASSOCIATION",
        "source_geometry": source["geometry"],
        "source_endpoints": rows,
        "input_checks": input_checks,
        "paired_analysis": primary,
        "negative_control": control,
        "passed": passed,
        "runtime_seconds": time.perf_counter() - started,
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "raw_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    checker = subprocess.run(
        [sys.executable, str(HERE / "independent_check.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(checker.stdout, end="" if checker.stdout.endswith("\n") else "\n")
    summary = {
        "status": result["scientific_status"],
        "verdict": result["claim_verdict"],
        "pairs": 5,
        "sw2_mean_difference": primary["sw2"]["mean"],
        "sw2_interval": primary["sw2"]["percentile_95_interval"],
        "msw2_mean_difference": primary["msw2"]["mean"],
        "msw2_interval": primary["msw2"]["percentile_95_interval"],
        "all_ten_endpoint_checks_passed": input_checks["endpoint_checkers_passed"],
        "negative_control_rejected": control["rejected"],
        "independent_checker_passed": checker.returncode == 0,
    }
    (ARTIFACT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_FIVE_PAIR_SUMMARY=" + json.dumps(summary, sort_keys=True))
    print("CLAIM_1_4_FIVE_PAIR=" + json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed and checker.returncode == 0 else 1)


if __name__ == "__main__":
    main()
