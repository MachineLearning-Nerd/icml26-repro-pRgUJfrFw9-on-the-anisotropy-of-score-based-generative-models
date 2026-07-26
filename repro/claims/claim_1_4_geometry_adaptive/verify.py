"""Variance-allocated estimator of the exact iDDPM average geometry."""

from __future__ import annotations

import base64
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.modules import Diffuser


HERE = Path(__file__).resolve().parent
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1_4_geometry_adaptive"
ARTIFACT.mkdir(parents=True, exist_ok=True)
SELECTED = [0, 31, 63, 95, 127, 159, 191, 223, 255]


def make_model():
    return Diffuser(
        shape=[1, 16, 16],
        T=1000,
        linear=True,
        model_cfg={
            "base_channels": 32,
            "num_res_attn_blocks": 1,
            "channel_mults": [1, 1],
            "is_attn": [False, False, False],
            "num_heads": 4,
            "dropout": 0.0,
            "use_scale_shift_norm": False,
            "downsample_with_pool": True,
            "interpolation": "nearest",
            "patch_size": 2,
        },
    )


def score_vector(seed, timestep_value):
    torch.manual_seed(seed)
    model = make_model().eval()
    timestep = torch.tensor([[[[timestep_value]]]], dtype=torch.int64)
    with torch.inference_mode():
        output = model.epsilon(torch.zeros(1, 1, 16, 16), timestep)
        score = output / model.sqrt_one_minus_alphas_cumprod[timestep.squeeze()]
    vector = score.reshape(-1).double().numpy()
    del model, output, score
    return vector


def pilot_proxies():
    proxies = np.empty(1000)
    norms = np.empty(1000)
    for timestep in range(1000):
        first = score_vector(200_000 + 2 * timestep, timestep)
        second = score_vector(200_001 + 2 * timestep, timestep)
        first_norm2 = float(first @ first)
        second_norm2 = float(second @ second)
        difference_squared = (
            first_norm2**2
            + second_norm2**2
            - 2.0 * float(first @ second) ** 2
        )
        difference = math.sqrt(max(difference_squared, 0.0)) / math.sqrt(2.0)
        mean_scale = 0.1 * (first_norm2 + second_norm2) / 2.0
        proxies[timestep] = max(difference, mean_scale, 1e-12)
        norms[timestep] = (first_norm2 + second_norm2) / 2.0
        if timestep % 100 == 99:
            gc.collect()
    return proxies, norms


def allocate_counts(proxies):
    extra_groups = 3000
    weights = proxies / proxies.sum()
    raw = weights * extra_groups
    groups = np.floor(raw).astype(int)
    remainder = extra_groups - int(groups.sum())
    order = np.argsort(raw - groups)[::-1]
    groups[order[:remainder]] += 1
    counts = 4 + 4 * groups
    assert int(counts.sum()) == 16000
    assert np.all(counts % 4 == 0)
    return counts


def estimate_quarters(counts):
    quarters = []
    seed_cursor = 300_000
    started = time.perf_counter()
    for quarter in range(4):
        geometry = np.zeros((256, 256), dtype=np.float64)
        for timestep in range(1000):
            conditional = np.zeros((256, 256), dtype=np.float64)
            count = int(counts[timestep] // 4)
            for _ in range(count):
                vector = score_vector(seed_cursor, timestep)
                seed_cursor += 1
                conditional += np.outer(vector, vector)
            geometry += conditional / count / 1000.0
            if timestep % 100 == 99:
                gc.collect()
        quarters.append(geometry)
        print(
            f"ADAPTIVE_GEOMETRY_PROGRESS quarter={quarter + 1}/4 "
            f"networks={sum(int(value // 4) for value in counts) * (quarter + 1)} "
            f"elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return np.asarray(quarters), seed_cursor, time.perf_counter() - started


def eig(matrix):
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def band_overlap(first, second, index, half_width=4):
    start = max(0, index - half_width)
    end = min(256, index + half_width + 1)
    singular = np.linalg.svd(first[:, start:end].T @ second[:, start:end], compute_uv=False)
    return {
        "start": start,
        "end_exclusive": end,
        "minimum_cosine": float(singular.min()),
        "mean_cosine": float(singular.mean()),
    }


def bootstrap(quarters):
    rng = np.random.default_rng(410_000)
    values = {str(index): [] for index in SELECTED}
    for _ in range(200):
        indices = rng.integers(0, 4, 4)
        spectrum = np.linalg.eigvalsh(quarters[indices].mean(axis=0))[::-1]
        for index in SELECTED:
            values[str(index)].append(float(spectrum[index]))
    return {
        key: {
            "lower": float(np.quantile(row, 0.025)),
            "median": float(np.median(row)),
            "upper": float(np.quantile(row, 0.975)),
        }
        for key, row in values.items()
    }


def load_independent():
    spec = importlib.util.spec_from_file_location("adaptive_independent", HERE / "independent_check.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def encode(matrix):
    raw = np.asarray(matrix, dtype="<f8").tobytes(order="C")
    return {
        "dtype": "<f8",
        "shape": [256, 256],
        "encoding": "base64(zlib(raw C-order bytes))",
        "sha256_raw": hashlib.sha256(raw).hexdigest(),
        "payload": base64.b64encode(zlib.compress(raw, 9)).decode("ascii"),
    }


def negative_control(matrix):
    rng = np.random.default_rng(420_000)
    residuals = []
    for _ in range(64):
        vector = rng.normal(size=256)
        vector /= np.linalg.norm(vector)
        rayleigh = float(vector @ matrix @ vector)
        residuals.append(
            float(np.linalg.norm(matrix @ vector - rayleigh * vector) / np.linalg.norm(matrix @ vector))
        )
    return {
        "expected_to_fail": True,
        "control_claim": "random directions are geometry eigenvectors",
        "median_relative_residual": float(np.median(residuals)),
        "rejected": bool(np.median(residuals) > 0.05),
    }


def main():
    started = time.perf_counter()
    proxies, pilot_norms = pilot_proxies()
    counts = allocate_counts(proxies)
    quarters, seed_end, estimator_seconds = estimate_quarters(counts)
    matrix = quarters.mean(axis=0)
    values, vectors = eig(matrix)
    first_values, first_vectors = eig(quarters[[0, 2]].mean(axis=0))
    second_values, second_vectors = eig(quarters[[1, 3]].mean(axis=0))
    split = []
    for index in SELECTED:
        split.append(
            {
                "index": index,
                "absolute_vector_overlap": float(abs(first_vectors[:, index] @ second_vectors[:, index])),
                "band_overlap": band_overlap(first_vectors, second_vectors, index),
                "first_eigenvalue": float(first_values[index]),
                "second_eigenvalue": float(second_values[index]),
            }
        )
    endpoint_rows = [split[0], split[-1]]
    ready = all(
        row["absolute_vector_overlap"] >= 0.85
        and row["band_overlap"]["mean_cosine"] >= 0.75
        for row in endpoint_rows
    )
    orthogonality = float(np.max(np.abs(vectors.T @ vectors - np.eye(256))))
    residuals = [
        float(
            np.linalg.norm(matrix @ vectors[:, index] - values[index] * vectors[:, index])
            / np.linalg.norm(matrix @ vectors[:, index])
        )
        for index in SELECTED
    ]
    independent = load_independent().run(matrix, values)
    control = negative_control(matrix)
    structural_pass = (
        float(values[-1]) >= -1e-9
        and orthogonality < 1e-10
        and max(residuals) < 1e-10
        and independent["passed"]
        and control["rejected"]
    )
    allocation_order = np.argsort(counts)[::-1][:20]
    result = {
        "status": "DIRECTIONS READY FOR TRAINING" if ready else "DIRECTION STABILITY INSUFFICIENT",
        "structural_pass": bool(structural_pass),
        "generation_direction_ready": bool(ready),
        "claims_1_and_4_verdict": "BLOCKED pending actual training and sampling",
        "pilot": {
            "networks": 2000,
            "seed_range": [200_000, 201_999],
            "conditional_norm_min": float(pilot_norms.min()),
            "conditional_norm_max": float(pilot_norms.max()),
        },
        "allocation": {
            "final_networks": int(counts.sum()),
            "minimum_per_timestep": int(counts.min()),
            "maximum_per_timestep": int(counts.max()),
            "top_twenty": [
                {"timestep": int(index), "networks": int(counts[index]), "proxy": float(proxies[index])}
                for index in allocation_order
            ],
            "counts": counts.tolist(),
        },
        "final_seed_range": [300_000, seed_end - 1],
        "selected_descending_indices": SELECTED,
        "selected_eigenvalues": {str(index): float(values[index]) for index in SELECTED},
        "selected_eigenvectors": {str(index): vectors[:, index].tolist() for index in SELECTED},
        "split_half_stability": split,
        "bootstrap_intervals": bootstrap(quarters),
        "matrix_audit": {
            "lambda_max": float(values[0]),
            "lambda_min": float(values[-1]),
            "condition_number": float(values[0] / values[-1]),
            "orthonormality_error": orthogonality,
            "selected_max_eigen_residual": max(residuals),
        },
        "independent_checker": independent,
        "negative_control": control,
        "estimator_runtime_seconds": estimator_seconds,
        "total_runtime_seconds": time.perf_counter() - started,
        "complete_geometry": encode(matrix),
        "gpu_used": False,
    }
    (ARTIFACT / "raw_adaptive_geometry.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CLAIM_1_4_ADAPTIVE_GEOMETRY=" + json.dumps(result, sort_keys=True))
    print(
        "CLAIM_1_4_ADAPTIVE_GEOMETRY_SUMMARY "
        f"structural_pass={structural_pass} ready={ready} "
        f"top_overlap={split[0]['absolute_vector_overlap']:.6f} "
        f"bottom_overlap={split[-1]['absolute_vector_overlap']:.6f} "
        f"lambda_max={values[0]:.9g} lambda_min={values[-1]:.9g} "
        f"allocation_max={counts.max()} runtime_seconds={estimator_seconds:.3f}"
    )
    raise SystemExit(0 if structural_pass else 1)


if __name__ == "__main__":
    main()
