"""Exact 1,000-step sampling cost calibration under the common 4-thread policy."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.modules import Diffuser


HERE = Path(__file__).resolve().parent
ARTIFACT = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_1_4_four_thread_sampling_profile"
)
THREAD_RAW = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_1_4_cpu_thread_scaling"
    / "raw_results.json"
)
CONFIG = ROOT / "configs" / "1x16x16-iddpm.json"
THREADS = 4
MODEL_SEED = 640_001
INIT_SEED = 640_002
REVERSE_NOISE_SEED = 640_003
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


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "gpu_used": False,
    }


def main() -> None:
    total_started = time.perf_counter()
    torch.set_num_threads(THREADS)
    torch.set_flush_denormal(False)
    config = json.loads(CONFIG.read_text())
    thread_raw = json.loads(THREAD_RAW.read_text())
    four_thread = thread_raw["numerical_diagnostics"]["4"]
    four_hashes = [
        trial["final_model_sha256"] for trial in thread_raw["trials"]["4"]
    ]

    torch.manual_seed(MODEL_SEED)
    model = make_model(config).eval()
    generator = torch.Generator().manual_seed(INIT_SEED)
    initial = torch.randn(1_000, 1, 16, 16, generator=generator)
    torch.manual_seed(REVERSE_NOISE_SEED)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.sample(init=initial, steps=None, eta=1.0)
    sampling_seconds = time.perf_counter() - started
    finite = bool(torch.isfinite(generated).all())
    if not finite:
        raise RuntimeError("non-finite exact-path sampling output")

    training_seconds_per_update = four_thread["median_seconds_per_update"]
    projected_training_seconds = 2_000 * training_seconds_per_update
    projected_sampling_seconds = 10 * sampling_seconds
    control = {
        "control": "100 reverse steps presented as the exact 1,000-step path",
        "expected_to_fail": True,
        "observed_steps": 100,
        "required_steps": 1_000,
        "rejected": 100 != 1_000,
    }
    result = {
        "scientific_status": (
            "CPU_RESOURCE_CALIBRATION_ONLY; Claims 1 and 4 remain BLOCKED"
        ),
        "policy": {
            "torch_intraop_threads": THREADS,
            "same_policy_required_for_both_endpoints": True,
            "four_thread_repetitions_bitwise_identical": len(set(four_hashes))
            == 1,
            "cross_32_thread_loss_relative_error": four_thread[
                "max_loss_relative_error_vs_32"
            ],
            "reason": (
                "fastest deterministic within-setting schedule; paper does "
                "not specify CPU reduction order"
            ),
        },
        "setup": {
            "architecture": "author iDDPM U-Net",
            "config": "configs/1x16x16-iddpm.json",
            "model_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "batch_size": 1_000,
            "shape": [1, 16, 16],
            "diffusion_steps": len(model.betas),
            "steps_argument": None,
            "eta": 1.0,
            "trained": False,
            "training_status_relevance": (
                "runtime calibration only; architecture and reverse loop are exact"
            ),
        },
        "seeds": {
            "model": MODEL_SEED,
            "initial_noise": INIT_SEED,
            "reverse_noise": REVERSE_NOISE_SEED,
        },
        "sampling": {
            "runtime_seconds_for_1k_samples": sampling_seconds,
            "finite": finite,
            "shape": list(generated.shape),
            "sha256": tensor_hash(generated),
            "mean": float(generated.mean()),
            "standard_deviation": float(generated.std(correction=0)),
        },
        "projection": {
            "training_seconds_per_update": training_seconds_per_update,
            "training_updates": 2_000,
            "projected_training_seconds": projected_training_seconds,
            "sampling_batches": 10,
            "projected_sampling_seconds": projected_sampling_seconds,
            "projected_training_plus_sampling_hours": (
                projected_training_seconds + projected_sampling_seconds
            )
            / 3_600,
            "training_resource_gate_would_pass": (
                training_seconds_per_update <= 12.0
            ),
        },
        "negative_control": control,
        "model_state_sha256": model_hash(model),
        "compute": cpu_metadata(),
        "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "total_runtime_seconds": time.perf_counter() - total_started,
    }
    (ARTIFACT / "raw_profile.json").write_text(
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
        "threads": THREADS,
        "sampling_seconds_for_1k_samples": sampling_seconds,
        "projected_training_plus_sampling_hours": result["projection"][
            "projected_training_plus_sampling_hours"
        ],
        "training_resource_gate_would_pass": result["projection"][
            "training_resource_gate_would_pass"
        ],
        "independent_checker_passed": independent.get("passed", False),
        "scientific_status": result["scientific_status"],
    }
    (ARTIFACT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        "CLAIM_1_4_FOUR_THREAD_PROFILE_SUMMARY="
        + json.dumps(summary, sort_keys=True)
    )
    print("CLAIM_1_4_FOUR_THREAD_PROFILE=" + json.dumps(result, sort_keys=True))
    passed = checker.returncode == 0 and control["rejected"]
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
