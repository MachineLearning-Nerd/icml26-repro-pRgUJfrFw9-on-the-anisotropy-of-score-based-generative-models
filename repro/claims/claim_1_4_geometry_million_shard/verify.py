"""One predeclared half of the paper-scale one-million-network geometry.

Two sibling nodes each estimate 500,000 fresh author iDDPM initializations.
Their equal-weight aggregate is the paper's one-million-network estimator.
This shard is durable partial evidence and never a claim verdict by itself.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.modules import Diffuser


HERE = Path(__file__).resolve().parent
ARTIFACT = ROOT / ".openresearch" / "artifacts" / HERE.name
ARTIFACT.mkdir(parents=True, exist_ok=True)


def emit(label: str, payload: dict) -> None:
    print(label + "=" + json.dumps(payload, sort_keys=True), flush=True)


def make_model() -> Diffuser:
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


def split_ranges(start: int, count: int, task_count: int) -> list[tuple[int, int, int]]:
    quotient, remainder = divmod(count, task_count)
    tasks = []
    cursor = start
    for task_index in range(task_count):
        size = quotient + (1 if task_index < remainder else 0)
        tasks.append((task_index, cursor, cursor + size))
        cursor += size
    assert cursor == start + count
    return tasks


def worker(task: tuple[int, int, int]) -> tuple:
    task_index, seed_start, seed_end = task
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    zero = torch.zeros(1, 1, 16, 16)
    parity_sums = np.zeros((2, 256, 256), dtype=np.float64)
    parity_counts = np.zeros(2, dtype=np.int64)
    timestep_histogram = np.zeros(1000, dtype=np.int64)
    norm_sum = 0.0
    norm_sq_sum = 0.0
    started = time.perf_counter()

    for offset, seed in enumerate(range(seed_start, seed_end)):
        torch.manual_seed(seed)
        model = make_model().eval()
        timestep = model.randint(batch_size=1, device=torch.device("cpu"))
        with torch.inference_mode():
            output = model.epsilon(zero, timestep)
            score = output / model.sqrt_one_minus_alphas_cumprod[timestep.squeeze()]
        vector = score.reshape(-1).double().numpy()
        parity = seed & 1
        parity_sums[parity] += np.outer(vector, vector)
        parity_counts[parity] += 1
        timestep_histogram[int(timestep.item())] += 1
        norm2 = float(vector @ vector)
        norm_sum += norm2
        norm_sq_sum += norm2 * norm2
        del model, output, score, vector
        if offset % 256 == 255:
            gc.collect()

    return (
        task_index,
        parity_sums,
        parity_counts,
        timestep_histogram,
        norm_sum,
        norm_sq_sum,
        time.perf_counter() - started,
    )


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


def negative_control(matrix: np.ndarray) -> dict:
    rng = np.random.default_rng(630_000)
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
        "control_claim": "random directions are geometry eigenvectors",
        "median_relative_residual": float(np.median(residuals)),
        "rejected": bool(np.median(residuals) > 0.05),
    }


def cpu_metadata(worker_count: int) -> dict:
    affinity = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "affinity_cpus": affinity,
        "worker_processes": worker_count,
        "threads_per_worker": 1,
        "gpu_used": False,
    }


def main() -> None:
    total_started = time.perf_counter()
    design = json.loads((HERE / "design.json").read_text())
    count = int(design["network_count"])
    worker_count = int(design["worker_count"])
    task_count = int(design["task_count"])
    selected = [int(value) for value in design["selected_indices"]]
    if count != 500_000:
        raise RuntimeError(f"expected a 500000-network shard, observed {count}")
    if worker_count != 32 or task_count != 128:
        raise RuntimeError("paper-scale shard requires 32 workers and 128 tasks")
    if selected != [0, 31, 63, 95, 127, 159, 191, 223, 255]:
        raise RuntimeError("selected direction indices changed")

    compute = cpu_metadata(worker_count)
    emit("CLAIM_1_4_MILLION_SHARD_COMPUTE", compute)
    if compute["affinity_cpus"] is not None and compute["affinity_cpus"] < worker_count:
        raise RuntimeError(
            f"need {worker_count} CPU-affinity slots, observed {compute['affinity_cpus']}"
        )

    tasks = split_ranges(int(design["seed_start"]), count, task_count)
    parity_sums = np.zeros((2, 256, 256), dtype=np.float64)
    parity_counts = np.zeros(2, dtype=np.int64)
    timestep_histogram = np.zeros(1000, dtype=np.int64)
    norm_sum = 0.0
    norm_sq_sum = 0.0
    task_runtimes = []
    estimator_started = time.perf_counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context
    ) as executor:
        for completed, result in enumerate(executor.map(worker, tasks), start=1):
            (
                task_index,
                task_parity_sums,
                task_parity_counts,
                task_histogram,
                task_norm_sum,
                task_norm_sq_sum,
                task_runtime,
            ) = result
            parity_sums += task_parity_sums
            parity_counts += task_parity_counts
            timestep_histogram += task_histogram
            norm_sum += task_norm_sum
            norm_sq_sum += task_norm_sq_sum
            task_runtimes.append(task_runtime)
            if completed == 1 or completed % 8 == 0 or completed == task_count:
                emit(
                    "CLAIM_1_4_MILLION_SHARD_PROGRESS",
                    {
                        "completed_tasks": completed,
                        "of": task_count,
                        "last_task_index": task_index,
                        "networks_merged": int(parity_counts.sum()),
                        "elapsed_seconds": time.perf_counter() - estimator_started,
                    },
                )

    if int(parity_counts.sum()) != count:
        raise RuntimeError(
            f"network accounting mismatch: {parity_counts.sum()} versus {count}"
        )
    if int(timestep_histogram.sum()) != count:
        raise RuntimeError("timestep histogram does not sum to the network count")

    half_matrices = np.asarray(
        [parity_sums[index] / parity_counts[index] for index in range(2)]
    )
    matrix = parity_sums.sum(axis=0) / count
    values, vectors = eig(matrix)
    half_values_0, half_vectors_0 = eig(half_matrices[0])
    half_values_1, half_vectors_1 = eig(half_matrices[1])
    split_stability = []
    for index in selected:
        split_stability.append(
            {
                "index": index,
                "absolute_vector_overlap": float(
                    abs(half_vectors_0[:, index] @ half_vectors_1[:, index])
                ),
                "band_overlap": band_overlap(
                    half_vectors_0, half_vectors_1, index
                ),
                "first_eigenvalue": float(half_values_0[index]),
                "second_eigenvalue": float(half_values_1[index]),
            }
        )

    orthogonality_error = float(
        np.max(np.abs(vectors.T @ vectors - np.eye(256)))
    )
    selected_residuals = {
        str(index): float(
            np.linalg.norm(matrix @ vectors[:, index] - values[index] * vectors[:, index])
            / np.linalg.norm(matrix @ vectors[:, index])
        )
        for index in selected
    }
    control = negative_control(matrix)
    histogram_expected = count / 1000.0
    histogram_chi_square = float(
        np.sum((timestep_histogram - histogram_expected) ** 2 / histogram_expected)
    )
    norm_mean = norm_sum / count
    norm_variance = max(norm_sq_sum / count - norm_mean**2, 0.0)
    structural_pass = bool(
        values[-1] >= -1e-9
        and orthogonality_error < 1e-10
        and max(selected_residuals.values()) < 1e-10
        and control["rejected"]
        and parity_counts.tolist() == [250_000, 250_000]
    )
    result = {
        "claims": [1, 4],
        "scientific_status": "PARTIAL_PAPER_SCALE_GEOMETRY_SHARD",
        "claim_verdict": "BLOCKED_PENDING_SECOND_SHARD_AND_GENERATION",
        "design": design,
        "author_probe": {
            "architecture": "author iDDPM U-Net",
            "shape": [1, 16, 16],
            "dimension": 256,
            "input": "zero tensor (sigma=0)",
            "timestep": "uniform model.randint over 1000 diffusion steps",
            "initialization": "fresh default-initialized network per sample",
            "estimator": "streamed mean of score outer products",
        },
        "compute": compute,
        "network_accounting": {
            "count": count,
            "seed_range": [
                int(design["seed_start"]),
                int(design["seed_start"]) + count - 1,
            ],
            "parity_half_counts": parity_counts.tolist(),
            "timestep_histogram": timestep_histogram.tolist(),
            "timestep_chi_square": histogram_chi_square,
            "score_norm_squared_mean": norm_mean,
            "score_norm_squared_standard_deviation": math.sqrt(norm_variance),
        },
        "selected_descending_indices": selected,
        "selected_eigenvalues": {
            str(index): float(values[index]) for index in selected
        },
        "selected_eigenvectors": {
            str(index): vectors[:, index].tolist() for index in selected
        },
        "split_half_stability": split_stability,
        "matrix_audit": {
            "lambda_max": float(values[0]),
            "lambda_min": float(values[-1]),
            "condition_number": float(values[0] / values[-1]),
            "orthonormality_error": orthogonality_error,
            "selected_max_eigen_residual": max(selected_residuals.values()),
            "selected_eigen_residuals": selected_residuals,
        },
        "negative_control": control,
        "structural_pass": structural_pass,
        "complete_geometry": encode(matrix),
        "parity_half_geometries": [encode(matrix_) for matrix_ in half_matrices],
        "runtime": {
            "estimator_seconds": time.perf_counter() - estimator_started,
            "total_seconds": time.perf_counter() - total_started,
            "task_min_seconds": float(np.min(task_runtimes)),
            "task_median_seconds": float(np.median(task_runtimes)),
            "task_max_seconds": float(np.max(task_runtimes)),
            "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        },
    }
    raw_path = ARTIFACT / "raw_results.json"
    raw_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    emit(
        "CLAIM_1_4_MILLION_SHARD_SUMMARY",
        {
            "shard": design["shard"],
            "networks": count,
            "lambda_max": result["matrix_audit"]["lambda_max"],
            "lambda_min": result["matrix_audit"]["lambda_min"],
            "interior_vector_overlaps": {
                str(row["index"]): row["absolute_vector_overlap"]
                for row in split_stability
            },
            "structural_pass": structural_pass,
            "estimator_seconds": result["runtime"]["estimator_seconds"],
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
    print(checker.stdout, end="" if checker.stdout.endswith("\n") else "\n", flush=True)
    emit("CLAIM_1_4_MILLION_SHARD", result)
    raise SystemExit(0 if structural_pass and checker.returncode == 0 else 1)


if __name__ == "__main__":
    main()
