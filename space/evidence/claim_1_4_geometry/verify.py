"""Convergence-audited iDDPM average-geometry estimator."""

from __future__ import annotations

import base64
import gc
import hashlib
import importlib.util
import json
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
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1_4_geometry"
ARTIFACT.mkdir(parents=True, exist_ok=True)
SELECTED = [0, 31, 63, 95, 127, 159, 191, 223, 255]
HORIZONS = [1000, 2000, 4000, 8000]


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


def load_independent_checker():
    spec = importlib.util.spec_from_file_location(
        "claim14_geometry_independent", HERE / "independent_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def eigendecompose(matrix):
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def timestep_permutation(block):
    rng = np.random.default_rng(90_000 + block)
    return rng.permutation(1000)


def estimate_geometry():
    dimension = 256
    block_matrices = []
    checkpoint_rows = []
    running = np.zeros((dimension, dimension), dtype=np.float64)
    started = time.perf_counter()
    for block in range(8):
        block_sum = np.zeros((dimension, dimension), dtype=np.float64)
        permutation = timestep_permutation(block)
        for within_block in range(1000):
            global_index = block * 1000 + within_block
            torch.manual_seed(100_000 + global_index)
            model = make_model().eval()
            timestep_value = int(permutation[within_block])
            timestep = torch.tensor([[[[timestep_value]]]], dtype=torch.int64)
            with torch.inference_mode():
                output = model.epsilon(torch.zeros(1, 1, 16, 16), timestep)
                score = output / model.sqrt_one_minus_alphas_cumprod[
                    timestep.squeeze()
                ]
            vector = score.reshape(-1).double().numpy()
            block_sum += np.outer(vector, vector)
            del model, output, score
            if within_block % 100 == 99:
                gc.collect()
        block_matrix = block_sum / 1000
        block_matrices.append(block_matrix)
        running += block_sum
        horizon = (block + 1) * 1000
        if horizon in HORIZONS:
            values, vectors = eigendecompose(running / horizon)
            checkpoint_rows.append(
                {
                    "horizon": horizon,
                    "eigenvalues": values,
                    "vectors": vectors,
                }
            )
        print(
            f"GEOMETRY_PROGRESS networks={horizon} "
            f"elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return np.asarray(block_matrices), checkpoint_rows, time.perf_counter() - started


def subspace_overlap(first_vectors, second_vectors, center, half_width=4):
    start = max(0, center - half_width)
    end = min(first_vectors.shape[1], center + half_width + 1)
    singular = np.linalg.svd(
        first_vectors[:, start:end].T @ second_vectors[:, start:end],
        compute_uv=False,
    )
    return {
        "index_start": start,
        "index_end_exclusive": end,
        "minimum_cosine": float(singular.min()),
        "mean_cosine": float(singular.mean()),
    }


def bootstrap_intervals(block_matrices):
    rng = np.random.default_rng(91_000)
    selected_values = {str(index): [] for index in SELECTED}
    for _ in range(200):
        indices = rng.integers(0, len(block_matrices), len(block_matrices))
        matrix = block_matrices[indices].mean(axis=0)
        values = np.linalg.eigvalsh(matrix)[::-1]
        for index in SELECTED:
            selected_values[str(index)].append(float(values[index]))
    return {
        index: {
            "lower_2_5_percent": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)),
            "upper_97_5_percent": float(np.quantile(values, 0.975)),
        }
        for index, values in selected_values.items()
    }


def negative_control(matrix, eigenvalues):
    rng = np.random.default_rng(92_000)
    residuals = []
    for _ in range(64):
        direction = rng.normal(size=matrix.shape[0])
        direction /= np.linalg.norm(direction)
        rayleigh = float(direction @ matrix @ direction)
        residual = np.linalg.norm(matrix @ direction - rayleigh * direction)
        residual /= max(np.linalg.norm(matrix @ direction), 1e-30)
        residuals.append(float(residual))
    rejected = float(np.median(residuals)) > 0.05
    return {
        "control_claim": "fixed random directions are eigenvectors of the estimated geometry",
        "expected_to_fail": True,
        "rejected": bool(rejected),
        "median_relative_eigen_residual": float(np.median(residuals)),
        "minimum_relative_eigen_residual": float(np.min(residuals)),
        "rayleigh_bounds": [float(eigenvalues[-1]), float(eigenvalues[0])],
    }


def encode_matrix(matrix):
    little_endian = np.asarray(matrix, dtype="<f8")
    raw = little_endian.tobytes(order="C")
    compressed = zlib.compress(raw, level=9)
    return {
        "dtype": "<f8",
        "shape": list(matrix.shape),
        "encoding": "base64(zlib(raw C-order bytes))",
        "sha256_raw": hashlib.sha256(raw).hexdigest(),
        "payload": base64.b64encode(compressed).decode("ascii"),
    }


def main():
    block_matrices, checkpoints, estimator_runtime = estimate_geometry()
    matrix = block_matrices.mean(axis=0)
    eigenvalues, eigenvectors = eigendecompose(matrix)
    final_checkpoint = checkpoints[-1]

    checkpoint_stability = []
    for checkpoint in checkpoints[:-1]:
        checkpoint_stability.append(
            {
                "horizon": checkpoint["horizon"],
                "selected_absolute_vector_overlaps": {
                    str(index): float(
                        abs(
                            checkpoint["vectors"][:, index]
                            @ final_checkpoint["vectors"][:, index]
                        )
                    )
                    for index in SELECTED
                },
            }
        )

    first_half = block_matrices[[0, 2, 4, 6]].mean(axis=0)
    second_half = block_matrices[[1, 3, 5, 7]].mean(axis=0)
    first_values, first_vectors = eigendecompose(first_half)
    second_values, second_vectors = eigendecompose(second_half)
    split_rows = []
    for index in SELECTED:
        split_rows.append(
            {
                "index": index,
                "first_eigenvalue": float(first_values[index]),
                "second_eigenvalue": float(second_values[index]),
                "relative_eigenvalue_difference": float(
                    abs(first_values[index] - second_values[index])
                    / max(abs(eigenvalues[index]), 1e-30)
                ),
                "absolute_vector_overlap": float(
                    abs(first_vectors[:, index] @ second_vectors[:, index])
                ),
                "band_subspace": subspace_overlap(
                    first_vectors, second_vectors, index
                ),
            }
        )

    orthonormality_error = float(
        np.max(np.abs(eigenvectors.T @ eigenvectors - np.eye(256)))
    )
    eigen_residuals = []
    for index in SELECTED:
        residual = np.linalg.norm(
            matrix @ eigenvectors[:, index]
            - eigenvalues[index] * eigenvectors[:, index]
        )
        residual /= max(np.linalg.norm(matrix @ eigenvectors[:, index]), 1e-30)
        eigen_residuals.append(float(residual))
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    minimum_eigenvalue = float(eigenvalues[-1])
    independent = load_independent_checker().run(matrix, eigenvalues)
    control = negative_control(matrix, eigenvalues)

    accepted = (
        symmetry_error < 1e-12
        and minimum_eigenvalue >= -1e-9
        and orthonormality_error < 1e-10
        and max(eigen_residuals) < 1e-10
        and independent["passed"]
        and control["rejected"]
    )
    result = {
        "status": "GEOMETRY_CALIBRATED; Claims 1 and 4 remain BLOCKED pending generation",
        "accepted": bool(accepted),
        "architecture": "author iDDPM U-Net, configs/1x16x16-iddpm.json",
        "probe": "x=0; each 1000-network block uses every timestep once",
        "network_initializations": 8000,
        "seed_range": [100_000, 107_999],
        "timestep_permutation_seeds": [90_000, 90_007],
        "selected_descending_indices": SELECTED,
        "selected_eigenvalues": {
            str(index): float(eigenvalues[index]) for index in SELECTED
        },
        "selected_eigenvectors": {
            str(index): eigenvectors[:, index].tolist() for index in SELECTED
        },
        "checkpoint_stability": checkpoint_stability,
        "split_half_stability": split_rows,
        "bootstrap_seed": 91_000,
        "bootstrap_eigenvalue_intervals": bootstrap_intervals(block_matrices),
        "matrix_audit": {
            "trace": float(np.trace(matrix)),
            "minimum_eigenvalue": minimum_eigenvalue,
            "maximum_eigenvalue": float(eigenvalues[0]),
            "condition_number": float(eigenvalues[0] / eigenvalues[-1]),
            "symmetry_max_abs_error": symmetry_error,
            "orthonormality_max_abs_error": orthonormality_error,
            "selected_max_relative_eigen_residual": max(eigen_residuals),
        },
        "independent_checker": independent,
        "negative_control": control,
        "estimator_runtime_seconds": estimator_runtime,
        "complete_geometry": encode_matrix(matrix),
        "gpu_used": False,
    }
    (ARTIFACT / "raw_geometry.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_GEOMETRY=" + json.dumps(result, sort_keys=True))
    print(
        "CLAIM_1_4_GEOMETRY_SUMMARY "
        f"accepted={accepted} networks=8000 "
        f"lambda_max={eigenvalues[0]:.9g} lambda_min={eigenvalues[-1]:.9g} "
        f"condition={eigenvalues[0] / eigenvalues[-1]:.6g} "
        f"max_eigen_residual={max(eigen_residuals):.3e} "
        f"random_control_rejected={control['rejected']} "
        f"runtime_seconds={estimator_runtime:.3f}"
    )
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
