"""Independent reconstruction of the five-pair endpoint aggregate."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARTIFACT = ROOT / ".openresearch" / "artifacts" / HERE.name


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    coordinate = probability * (len(values) - 1)
    left = math.floor(coordinate)
    right = math.ceil(coordinate)
    if left == right:
        return values[left]
    weight = coordinate - left
    return values[left] + weight * (values[right] - values[left])


def main() -> None:
    source = json.loads((HERE / "inputs" / "endpoints.json").read_text())
    raw = json.loads((ARTIFACT / "raw_results.json").read_text())
    table = {
        (row["seed_index"], row["endpoint"]): row for row in source["endpoints"]
    }
    checks = {}
    recomputed = {}
    for metric in ("sw2", "msw2"):
        differences = [
            table[(seed, "largest")][metric] - table[(seed, "smallest")][metric]
            for seed in range(5)
        ]
        bootstrap = [
            sum(differences[index] for index in indices) / 5
            for indices in itertools.product(range(5), repeat=5)
        ]
        interval = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]
        recorded = raw["paired_analysis"][metric]
        recomputed[metric] = {
            "differences": differences,
            "mean": sum(differences) / 5,
            "interval": interval,
        }
        checks[f"{metric}_differences_exact"] = differences == recorded[
            "paired_differences"
        ]
        checks[f"{metric}_mean_match"] = abs(
            recomputed[metric]["mean"] - recorded["mean"]
        ) < 1e-12
        checks[f"{metric}_interval_match"] = max(
            abs(interval[index] - recorded["percentile_95_interval"][index])
            for index in (0, 1)
        ) < 1e-12
        checks[f"{metric}_interval_positive"] = interval[0] > 0
    checks["all_run_ids_unique"] = len(
        {row["run_id"] for row in source["endpoints"]}
    ) == 10
    checks["endpoint_flags_pass"] = all(
        row["exact_setup_passed"] and row["independent_checker_passed"]
        for row in source["endpoints"]
    )
    checks["reverse_label_control_rejected"] = raw["negative_control"]["rejected"] is True
    checks["downscaled_status_honest"] = (
        raw["scientific_status"] == "HISTORICAL_DOWNSCALED_CORROBORATION"
        and raw["claim_verdict"]
        == "BLOCKED_PENDING_PAPER_SCALE_GENERATION_AND_RANK_ASSOCIATION"
    )
    passed = all(checks.values())
    output = {
        "implementation": "independent input parse and exhaustive stdlib bootstrap; no primary-verifier imports",
        "checks": checks,
        "recomputed": recomputed,
        "passed": passed,
    }
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_FIVE_PAIR_INDEPENDENT=" + json.dumps(output, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
