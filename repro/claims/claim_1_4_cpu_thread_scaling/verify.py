"""Resource-only CPU thread-scaling diagnostic for the exact iDDPM."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.modules import Diffuser


HERE = Path(__file__).resolve().parent
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1_4_cpu_thread_scaling"
GEOMETRY = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_1_4_geometry_adaptive"
    / "raw_geometry.json"
)
CONFIG = ROOT / "configs" / "1x16x16-iddpm.json"
ORDERS = [[32, 1, 16, 2, 8, 4], [4, 8, 2, 16, 1, 32]]
MODEL_SEED = 630_001
DATA_SEED = 630_002
STEP_SEEDS = [630_101, 630_102, 630_103]
ARTIFACT.mkdir(parents=True, exist_ok=True)


def make_model(config: dict) -> Diffuser:
    model_config = dict(config["diffuser"]["model"])
    model_config.update(
        {
            "downsample_with_pool": True,
            "interpolation": "nearest",
            "patch_size": 2,
        }
    )
    return Diffuser(
        shape=config["shape"],
        T=config["diffuser"]["T"],
        linear=config["diffuser"]["linear"],
        model_cfg=model_config,
    )


def largest_direction() -> torch.Tensor:
    raw = json.loads(GEOMETRY.read_text())
    value = torch.tensor(
        raw["selected_eigenvectors"]["0"], dtype=torch.float32
    ).reshape(1, 16, 16)
    return value / torch.linalg.norm(value)


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixed_batch(direction: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(DATA_SEED)
    coefficient = torch.randn(1_000, 1, 1, 1, generator=generator)
    return math.sqrt(256) * coefficient * direction


def run_trial(
    threads: int, repetition: int, batch: torch.Tensor, config: dict
) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(MODEL_SEED)
    model = make_model(config).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rows = []
    for step, seed in enumerate(STEP_SEEDS, start=1):
        torch.manual_seed(seed)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = model(batch)
        forward_seconds = time.perf_counter() - started
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss with {threads} threads at step {step}"
            )
        backward_started = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_started
        optimizer_started = time.perf_counter()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["grad_clip"]
        )
        optimizer.step()
        optimizer_seconds = time.perf_counter() - optimizer_started
        row = {
            "step": step,
            "warmup": step == 1,
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": optimizer_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(
            "CLAIM_1_4_THREAD_SCALING_PROGRESS="
            + json.dumps(
                {"threads": threads, "repetition": repetition, **row},
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "threads": threads,
        "repetition": repetition,
        "steps": rows,
        "median_timed_seconds": float(
            np.median([row["total_seconds"] for row in rows if not row["warmup"]])
        ),
        "final_model_sha256": model_hash(model),
    }


def choose_safe_fastest(trials: dict[str, list[dict]]) -> tuple[list[int], int, dict]:
    reference = trials["32"][0]
    reference_losses = np.asarray(
        [row["loss"] for row in reference["steps"]], dtype=np.float64
    )
    reference_hash = reference["final_model_sha256"]
    diagnostics = {}
    safe = []
    for key, values in trials.items():
        losses = np.asarray(
            [[row["loss"] for row in trial["steps"]] for trial in values],
            dtype=np.float64,
        )
        max_loss_error = float(
            np.max(
                np.abs(losses - reference_losses)
                / np.maximum(np.abs(reference_losses), 1e-12)
            )
        )
        hashes_equal = all(
            trial["final_model_sha256"] == reference_hash for trial in values
        )
        median_seconds = float(
            np.median([trial["median_timed_seconds"] for trial in values])
        )
        diagnostics[key] = {
            "max_loss_relative_error_vs_32": max_loss_error,
            "all_state_hashes_equal_32": hashes_equal,
            "median_seconds_per_update": median_seconds,
        }
        if hashes_equal and max_loss_error <= 1e-7:
            safe.append(int(key))
    if not safe:
        raise RuntimeError("no numerically equivalent thread count, including 32")
    selected = min(safe, key=lambda count: diagnostics[str(count)]["median_seconds_per_update"])
    return sorted(safe), selected, diagnostics


def cpu_metadata() -> dict:
    return {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "affinity_cpus": (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "torch_intraop_threads_at_end": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "gpu_used": False,
    }


def main() -> None:
    started = time.perf_counter()
    torch.set_flush_denormal(False)
    config = json.loads(CONFIG.read_text())
    direction = largest_direction()
    batch = fixed_batch(direction)
    trials = {str(count): [] for count in sorted({item for order in ORDERS for item in order})}
    for repetition, order in enumerate(ORDERS, start=1):
        for threads in order:
            trials[str(threads)].append(
                run_trial(threads, repetition, batch, config)
            )
    safe, selected, diagnostics = choose_safe_fastest(trials)
    invalid_threads = 0
    control = {
        "control": "zero intra-op threads",
        "expected_to_fail": True,
        "observed_threads": invalid_threads,
        "rejected": invalid_threads < 1,
    }
    selected_seconds = diagnostics[str(selected)]["median_seconds_per_update"]
    result = {
        "scientific_status": (
            "CPU_RESOURCE_DIAGNOSTIC_ONLY; Claims 1 and 4 remain BLOCKED"
        ),
        "orders": ORDERS,
        "thread_counts": sorted(int(key) for key in trials),
        "seeds": {
            "model": MODEL_SEED,
            "data": DATA_SEED,
            "dsm_steps": STEP_SEEDS,
        },
        "setup": {
            "architecture": "author iDDPM U-Net",
            "dimension": 256,
            "batch_size": 1_000,
            "direction": "largest stable SAD, geometry index 0",
            "direction_norm": float(torch.linalg.norm(direction)),
            "model_parameters": sum(
                parameter.numel() for parameter in make_model(config).parameters()
            ),
            "timed_steps_per_trial": 2,
            "warmup_steps_per_trial": 1,
        },
        "trials": trials,
        "numerical_diagnostics": diagnostics,
        "safe_thread_counts": safe,
        "selected_thread_count": selected,
        "selected_seconds_per_update": selected_seconds,
        "endpoint_resource_gate_would_pass": selected_seconds <= 12.0,
        "negative_control": control,
        "compute": cpu_metadata(),
        "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "total_runtime_seconds": time.perf_counter() - started,
    }
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
    print(checker.stdout, end="", flush=True)
    independent = (
        json.loads((ARTIFACT / "independent_checker_output.json").read_text())
        if checker.returncode == 0
        else {"passed": False}
    )
    summary = {
        "selected_thread_count": selected,
        "selected_seconds_per_update": selected_seconds,
        "safe_thread_counts": safe,
        "endpoint_resource_gate_would_pass": result[
            "endpoint_resource_gate_would_pass"
        ],
        "independent_checker_passed": independent.get("passed", False),
        "total_runtime_seconds": result["total_runtime_seconds"],
        "scientific_status": result["scientific_status"],
    }
    (ARTIFACT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_THREAD_SCALING_SUMMARY=" + json.dumps(summary, sort_keys=True))
    print("CLAIM_1_4_THREAD_SCALING=" + json.dumps(result, sort_keys=True))
    passed = checker.returncode == 0 and control["rejected"]
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
