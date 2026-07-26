"""Fixed experiment entrypoint shared by every node in the campaign."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


def cpu_metadata():
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "affinity_cpus": affinity,
        "machine": platform.machine(),
    }


def run_checker(checker):
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, str(checker)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    print(f"\nCHECKER={checker.relative_to(ROOT)}", flush=True)
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    elapsed = time.perf_counter() - started
    print(f"CHECKER_EXIT={return_code} CHECKER_RUNTIME_SECONDS={elapsed:.6f}")
    return {
        "checker": str(checker.relative_to(ROOT)),
        "exit_code": return_code,
        "runtime_seconds": elapsed,
    }


def main():
    started = time.perf_counter()
    checkers = [ROOT / "repro" / "src" / "verify_sad.py"]
    frozen_calibrations = {
        "claim_1_4_profile",
        "claim_1_4_geometry",
        "claim_1_4_geometry_adaptive",
        "claim_1_4_cpu_value_diagnostic",
        "claim_1_4_generation_endpoint",
    }
    checkers.extend(
        checker
        for checker in sorted((ROOT / "repro" / "claims").glob("*/verify.py"))
        if checker.parent.name not in frozen_calibrations
    )

    print("CAMPAIGN_FIXED_COMMAND=uv sync --locked && .venv/bin/python repro/run_campaign.py")
    print("CPU_METADATA=" + json.dumps(cpu_metadata(), sort_keys=True))
    results = [run_checker(checker) for checker in checkers]
    summary = {
        "cpu": cpu_metadata(),
        "runtime_seconds": time.perf_counter() - started,
        "checkers": results,
        "all_passed": all(result["exit_code"] == 0 for result in results),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "campaign_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("CAMPAIGN_SUMMARY=" + json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
