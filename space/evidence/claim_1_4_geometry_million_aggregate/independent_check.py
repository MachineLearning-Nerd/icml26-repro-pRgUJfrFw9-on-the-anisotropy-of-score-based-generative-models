"""Independent reconstruction of the one-million-network aggregate."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
ARTIFACT = ROOT / ".openresearch" / "artifacts" / HERE.name


def decode(record: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(record["payload"]))
    if hashlib.sha256(raw).hexdigest() != record["sha256_raw"]:
        raise RuntimeError("payload hash mismatch")
    return np.frombuffer(raw, dtype="<f8").reshape(record["shape"]).copy()


def main() -> None:
    shard_a = json.loads((INPUTS / "shard_a.json").read_text())
    shard_b = json.loads((INPUTS / "shard_b.json").read_text())
    raw = json.loads((ARTIFACT / "raw_results.json").read_text())
    reconstructed = (
        decode(shard_a["complete_geometry"])
        + decode(shard_b["complete_geometry"])
    ) / 2.0
    stored = decode(raw["complete_geometry"])
    values, vectors = np.linalg.eigh(reconstructed)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    selected = raw["selected_descending_indices"]
    eigenvalue_error = max(
        abs(values[index] - raw["selected_eigenvalues"][str(index)])
        for index in selected
    )
    checks = {
        "exact_network_count": raw["network_count"] == 1_000_000,
        "disjoint_seed_ranges": raw["source_shards"]["A"]["seed_range"][1]
        < raw["source_shards"]["B"]["seed_range"][0],
        "source_network_counts": raw["source_shards"]["A"]["networks"] == 500_000
        and raw["source_shards"]["B"]["networks"] == 500_000,
        "matrix_reconstructed": bool(np.max(np.abs(reconstructed - stored)) == 0),
        "matrix_symmetric": bool(
            np.max(np.abs(reconstructed - reconstructed.T)) < 1e-12
        ),
        "positive_semidefinite": bool(values[-1] >= -1e-9),
        "orthonormal": bool(
            np.max(np.abs(vectors.T @ vectors - np.eye(256))) < 1e-10
        ),
        "selected_eigenvalues_match": bool(eigenvalue_error < 1e-10),
        "negative_control_rejected": raw["negative_control"]["rejected"] is True,
        "structural_pass": raw["structural_pass"] is True,
        "no_gpu": raw["compute"]["gpu_used"] is False,
    }
    passed = all(checks.values())
    output = {
        "implementation": "independent shard decode, equal-weight matrix reconstruction, and NumPy eigensolver; no primary-verifier imports",
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
        "CLAIM_1_4_MILLION_AGGREGATE_INDEPENDENT="
        + json.dumps(output, sort_keys=True)
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
