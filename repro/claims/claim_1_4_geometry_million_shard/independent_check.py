"""Independent decoder and spectral audit for a 500k geometry shard."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / Path(__file__).resolve().parent.name


def decode(record: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(record["payload"]))
    if hashlib.sha256(raw).hexdigest() != record["sha256_raw"]:
        raise RuntimeError("geometry payload hash mismatch")
    return np.frombuffer(raw, dtype="<f8").reshape(record["shape"]).copy()


def main() -> None:
    raw = json.loads((ARTIFACT / "raw_results.json").read_text())
    matrix = decode(raw["complete_geometry"])
    halves = [decode(record) for record in raw["parity_half_geometries"]]
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    selected = raw["selected_descending_indices"]
    eigenvalue_error = max(
        abs(values[index] - raw["selected_eigenvalues"][str(index)])
        for index in selected
    )
    checks = {
        "network_count": raw["network_accounting"]["count"] == 500_000,
        "half_counts": raw["network_accounting"]["parity_half_counts"]
        == [250_000, 250_000],
        "timestep_count": sum(raw["network_accounting"]["timestep_histogram"])
        == 500_000,
        "matrix_symmetric": bool(np.max(np.abs(matrix - matrix.T)) < 1e-12),
        "half_average_matches": bool(
            np.max(np.abs((halves[0] + halves[1]) / 2.0 - matrix)) < 1e-12
        ),
        "positive_semidefinite": bool(values[-1] >= -1e-9),
        "orthonormal": bool(
            np.max(np.abs(vectors.T @ vectors - np.eye(256))) < 1e-10
        ),
        "selected_eigenvalues_match": bool(eigenvalue_error < 1e-10),
        "negative_control_rejected": raw["negative_control"]["rejected"] is True,
        "no_gpu": raw["compute"]["gpu_used"] is False,
        "worker_contract": raw["compute"]["worker_processes"] == 32
        and raw["compute"]["threads_per_worker"] == 1,
        "primary_structural_pass": raw["structural_pass"] is True,
    }
    passed = all(checks.values())
    output = {
        "implementation": "independent payload decode plus NumPy eigensolver; no primary-verifier imports",
        "checks": checks,
        "eigenvalue_max_abs_error": float(eigenvalue_error),
        "lambda_max": float(values[0]),
        "lambda_min": float(values[-1]),
        "passed": passed,
    }
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(
        "CLAIM_1_4_MILLION_SHARD_INDEPENDENT="
        + json.dumps(output, sort_keys=True)
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
