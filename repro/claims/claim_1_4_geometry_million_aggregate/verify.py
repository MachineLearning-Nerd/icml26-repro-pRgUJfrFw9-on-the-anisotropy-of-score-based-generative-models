"""Aggregate two frozen 500k shards into the paper-scale iDDPM geometry."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import zlib
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
ARTIFACT = ROOT / ".openresearch" / "artifacts" / HERE.name
ARTIFACT.mkdir(parents=True, exist_ok=True)
SELECTED = [0, 31, 63, 95, 127, 159, 191, 223, 255]
EXPECTED = {
    "A": {
        "run": "9027461d-3dae-46b2-b88d-cef5ed0592a0",
        "commit": "546bff1cfe9a1f5f931e0188761b795a106af324",
        "seed_range": [700_000, 1_199_999],
    },
    "B": {
        "run": "d7ea7073-d087-4dc1-9268-969c08b31123",
        "commit": "b6d44458d2062f7169459055bc9d906b44d0efe8",
        "seed_range": [1_200_000, 1_699_999],
    },
}


def emit(label: str, payload: dict) -> None:
    print(label + "=" + json.dumps(payload, sort_keys=True), flush=True)


def decode(record: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(record["payload"]))
    if hashlib.sha256(raw).hexdigest() != record["sha256_raw"]:
        raise RuntimeError("geometry payload hash mismatch")
    return np.frombuffer(raw, dtype="<f8").reshape(record["shape"]).copy()


def encode(matrix: np.ndarray) -> dict:
    raw = np.asarray(matrix, dtype="<f8").tobytes(order="C")
    return {
        "dtype": "<f8",
        "shape": [256, 256],
        "encoding": "base64(zlib(raw C-order bytes))",
        "sha256_raw": hashlib.sha256(raw).hexdigest(),
        "payload": base64.b64encode(zlib.compress(raw, 9)).decode("ascii"),
    }


def eig(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def band_overlap(
    first: np.ndarray, second: np.ndarray, index: int, half_width: int = 4
) -> dict:
    start = max(0, index - half_width)
    end = min(256, index + half_width + 1)
    singular = np.linalg.svd(
        first[:, start:end].T @ second[:, start:end], compute_uv=False
    )
    return {
        "start": start,
        "end_exclusive": end,
        "minimum_cosine": float(singular.min()),
        "mean_cosine": float(singular.mean()),
    }


def bootstrap(quarters: np.ndarray) -> dict:
    rng = np.random.default_rng(1_800_000)
    rows = {str(index): [] for index in SELECTED}
    for _ in range(1000):
        sampled = rng.integers(0, len(quarters), len(quarters))
        values = np.linalg.eigvalsh(quarters[sampled].mean(axis=0))[::-1]
        for index in SELECTED:
            rows[str(index)].append(float(values[index]))
    return {
        key: {
            "lower": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for key, values in rows.items()
    }


def negative_control(matrix: np.ndarray) -> dict:
    rng = np.random.default_rng(1_810_000)
    residuals = []
    for _ in range(64):
        vector = rng.normal(size=256)
        vector /= np.linalg.norm(vector)
        rayleigh = float(vector @ matrix @ vector)
        residuals.append(
            float(
                np.linalg.norm(matrix @ vector - rayleigh * vector)
                / np.linalg.norm(matrix @ vector)
            )
        )
    return {
        "expected_to_fail": True,
        "control_claim": "random directions are aggregate-geometry eigenvectors",
        "median_relative_residual": float(np.median(residuals)),
        "rejected": bool(np.median(residuals) > 0.05),
    }


def validate_input(record: dict, label: str) -> dict:
    expected = EXPECTED[label]
    checks = {
        "run_id": record["source_run_id"] == expected["run"],
        "commit": record["source_commit"] == expected["commit"],
        "label": record["design"]["shard"] == label,
        "network_count": record["network_accounting"]["count"] == 500_000,
        "seed_range": record["network_accounting"]["seed_range"]
        == expected["seed_range"],
        "half_counts": record["network_accounting"]["parity_half_counts"]
        == [250_000, 250_000],
        "timestep_count": sum(record["network_accounting"]["timestep_histogram"])
        == 500_000,
        "structural_pass": record["structural_pass"] is True,
        "negative_control": record["negative_control"]["rejected"] is True,
        "no_gpu": record["compute"]["gpu_used"] is False,
        "actual_cpu_allocation": record["compute"]["logical_cpus"] == 64
        and record["compute"]["physical_cpus"] == 32,
        "worker_contract": record["compute"]["worker_processes"] == 32
        and record["compute"]["threads_per_worker"] == 1,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    started = time.perf_counter()
    shard_a = json.loads((INPUTS / "shard_a.json").read_text())
    shard_b = json.loads((INPUTS / "shard_b.json").read_text())
    input_audit = {
        "A": validate_input(shard_a, "A"),
        "B": validate_input(shard_b, "B"),
    }
    if not all(row["passed"] for row in input_audit.values()):
        raise RuntimeError("frozen shard input audit failed")

    matrix_a = decode(shard_a["complete_geometry"])
    matrix_b = decode(shard_b["complete_geometry"])
    if EXPECTED["A"]["seed_range"][1] >= EXPECTED["B"]["seed_range"][0]:
        raise RuntimeError("shard seed ranges overlap")
    matrix = (matrix_a + matrix_b) / 2.0
    values, vectors = eig(matrix)
    values_a, vectors_a = eig(matrix_a)
    values_b, vectors_b = eig(matrix_b)
    quarters = np.asarray(
        [
            decode(record)
            for record in shard_a["parity_half_geometries"]
            + shard_b["parity_half_geometries"]
        ]
    )

    cross_shard_stability = []
    for index in SELECTED:
        vector_overlap = float(abs(vectors_a[:, index] @ vectors_b[:, index]))
        band = band_overlap(vectors_a, vectors_b, index)
        cross_shard_stability.append(
            {
                "index": index,
                "absolute_vector_overlap": vector_overlap,
                "band_overlap": band,
                "shard_a_eigenvalue": float(values_a[index]),
                "shard_b_eigenvalue": float(values_b[index]),
                "individual_direction_ready": bool(
                    vector_overlap >= 0.85 and band["mean_cosine"] >= 0.75
                ),
            }
        )

    orthogonality_error = float(
        np.max(np.abs(vectors.T @ vectors - np.eye(256)))
    )
    residuals = {
        str(index): float(
            np.linalg.norm(matrix @ vectors[:, index] - values[index] * vectors[:, index])
            / np.linalg.norm(matrix @ vectors[:, index])
        )
        for index in SELECTED
    }
    control = negative_control(matrix)
    stable_indices = [
        row["index"]
        for row in cross_shard_stability
        if row["individual_direction_ready"]
    ]
    structural_pass = bool(
        values[-1] >= -1e-9
        and orthogonality_error < 1e-10
        and max(residuals.values()) < 1e-10
        and control["rejected"]
        and all(row["passed"] for row in input_audit.values())
    )
    compute = {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "aggregation_threads": 1,
        "gpu_used": False,
        "source_shard_cpu": {
            "A": shard_a["compute"],
            "B": shard_b["compute"],
        },
    }
    result = {
        "claims": [1, 4],
        "scientific_status": "PAPER_SCALE_GEOMETRY_COMPLETE",
        "claim_verdict": "BLOCKED_PENDING_GENERATION_EXPERIMENTS",
        "network_count": 1_000_000,
        "source_shards": {
            "A": {
                "run_id": shard_a["source_run_id"],
                "commit": shard_a["source_commit"],
                "result_sha256": shard_a["source_result_sha256"],
                "networks": 500_000,
                "seed_range": EXPECTED["A"]["seed_range"],
            },
            "B": {
                "run_id": shard_b["source_run_id"],
                "commit": shard_b["source_commit"],
                "result_sha256": shard_b["source_result_sha256"],
                "networks": 500_000,
                "seed_range": EXPECTED["B"]["seed_range"],
            },
        },
        "input_audit": input_audit,
        "estimator": "equal-weight mean of two disjoint 500000-network shard means",
        "selected_descending_indices": SELECTED,
        "selected_eigenvalues": {
            str(index): float(values[index]) for index in SELECTED
        },
        "selected_eigenvectors": {
            str(index): vectors[:, index].tolist() for index in SELECTED
        },
        "cross_shard_stability": cross_shard_stability,
        "individual_direction_ready_indices": stable_indices,
        "bootstrap_eigenvalue_intervals": bootstrap(quarters),
        "matrix_audit": {
            "lambda_max": float(values[0]),
            "lambda_min": float(values[-1]),
            "condition_number": float(values[0] / values[-1]),
            "orthonormality_error": orthogonality_error,
            "selected_eigen_residuals": residuals,
            "selected_max_eigen_residual": max(residuals.values()),
        },
        "negative_control": control,
        "structural_pass": structural_pass,
        "complete_geometry": encode(matrix),
        "quarter_geometries": [encode(matrix_) for matrix_ in quarters],
        "compute": compute,
        "runtime_seconds": time.perf_counter() - started,
    }
    raw_path = ARTIFACT / "raw_results.json"
    raw_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    emit(
        "CLAIM_1_4_MILLION_AGGREGATE_SUMMARY",
        {
            "networks": 1_000_000,
            "lambda_max": result["matrix_audit"]["lambda_max"],
            "lambda_min": result["matrix_audit"]["lambda_min"],
            "stable_indices": stable_indices,
            "cross_shard_vector_overlaps": {
                str(row["index"]): row["absolute_vector_overlap"]
                for row in cross_shard_stability
            },
            "structural_pass": structural_pass,
            "runtime_seconds": result["runtime_seconds"],
        },
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
    emit("CLAIM_1_4_MILLION_AGGREGATE", result)
    raise SystemExit(0 if structural_pass and checker.returncode == 0 else 1)


if __name__ == "__main__":
    main()
