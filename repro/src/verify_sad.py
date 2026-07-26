"""Reproduce the exact judged Space's historical toy/proxy checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sad as S


OUT = Path(".openresearch/artifacts/historical_baseline")
OUT.mkdir(parents=True, exist_ok=True)
results = {}
rng = np.random.default_rng(42)
d = 6
sigma = 1.0


def record(name, passed, **values):
    results[name] = {"passed": bool(passed), **values}
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — HISTORICAL TOY/PROXY")


A = rng.standard_normal((d, d))
geometry = S.average_geometry_matrix(lambda x, _: A @ x, None, sigma, d, 2000, rng)
eigenvalues, sads = S.compute_sads(geometry)
orthogonality_error = float(np.max(np.abs(sads.T @ sads - np.eye(d))))
record(
    "claim_1",
    orthogonality_error < 1e-8 and bool(np.all(np.diff(np.abs(eigenvalues)) >= -1e-10)),
    orthogonality_error=orthogonality_error,
    eigenvalues=eigenvalues.tolist(),
)

test_eigenvalues = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
rates = S.convergence_rate_dsm(test_eigenvalues, sigma, 5)
record("claim_2", bool(np.all(np.diff(rates) <= 1e-10)), rates=rates.tolist())

ordered = np.sort(np.abs(eigenvalues))
alpha_identity = S.alignment(np.eye(d), sads, ordered)
alpha_reversed = S.alignment(np.fliplr(sads), sads, ordered)
record(
    "claim_3",
    alpha_reversed < alpha_identity * 0.5,
    alpha_identity=alpha_identity,
    alpha_reversed=alpha_reversed,
)

mlp_values = np.linalg.eigvalsh(S.mlp_geometry_matrix(d, 1.0, 2.0))
cnn_values = np.linalg.eigvalsh(S.cnn_geometry_matrix(d, 3))
transformer_values = np.linalg.eigvalsh(S.transformer_geometry_matrix(3, d))
record(
    "claim_6",
    bool(
        np.isclose(mlp_values[0], 1.0)
        and np.isclose(mlp_values[-1], 13.0)
        and len(np.unique(np.round(cnn_values, 4))) <= d
        and len(np.unique(np.round(transformer_values, 4))) <= 4
    ),
    mlp_eigenvalues=mlp_values.tolist(),
    cnn_eigenvalues=cnn_values.tolist(),
    transformer_eigenvalues=transformer_values.tolist(),
)

claim4_values = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
claim4_proxy = 1.0 - S.convergence_rate_dsm(claim4_values, sigma, 3)
record("claim_4", bool(np.all(np.diff(claim4_proxy) >= -1e-10)), proxy=claim4_proxy.tolist())

alignment_values = np.linspace(0.3, 1.0, 5) * alpha_identity
quality_proxy = 1.0 / np.maximum(alignment_values, 0.01)
record(
    "claim_5",
    bool(quality_proxy[-1] < quality_proxy[0]),
    alignment=alignment_values.tolist(),
    quality_proxy=quality_proxy.tolist(),
)

passed = all(item["passed"] for item in results.values())
(OUT / "raw_results.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"HISTORICAL_BASELINE_SUMMARY passed={sum(v['passed'] for v in results.values())}/6")
print("SCIENTIFIC_STATUS=Historical rejected baseline; no claim is VERIFIED")
raise SystemExit(0 if passed else 1)
