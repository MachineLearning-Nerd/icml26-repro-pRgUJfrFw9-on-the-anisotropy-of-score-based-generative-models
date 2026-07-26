"""Same-host CPU diagnostic for direction-dependent iDDPM runtime."""

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
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1_4_cpu_value_diagnostic"
GEOMETRY = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_1_4_geometry_adaptive"
    / "raw_geometry.json"
)
CONFIG = ROOT / "configs" / "1x16x16-iddpm.json"
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


def direction(index: int) -> torch.Tensor:
    raw = json.loads(GEOMETRY.read_text())
    value = torch.tensor(
        raw["selected_eigenvectors"][str(index)], dtype=torch.float32
    ).reshape(1, 16, 16)
    return value / torch.linalg.norm(value)


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batch_for(value: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(620_002)
    coefficient = torch.randn(1_000, 1, 1, 1, generator=generator)
    return math.sqrt(256) * coefficient * value


def run_condition(
    label: str, value: torch.Tensor, flush_denormal: bool, config: dict
) -> dict:
    supported = torch.set_flush_denormal(flush_denormal)
    if not supported:
        raise RuntimeError("CPU does not support torch.set_flush_denormal")
    torch.manual_seed(620_001)
    model = make_model(config).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    batch = batch_for(value)
    steps = []
    for step in range(1, 3):
        torch.manual_seed(620_100 + step)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = model(batch)
        forward_seconds = time.perf_counter() - started
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss in {label} step {step}")
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
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": optimizer_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        steps.append(row)
        print(
            "CLAIM_1_4_CPU_DIAGNOSTIC_PROGRESS="
            + json.dumps({"condition": label, **row}, sort_keys=True),
            flush=True,
        )
    return {
        "label": label,
        "flush_denormal": flush_denormal,
        "direction_norm": float(torch.linalg.norm(value)),
        "steps": steps,
        "median_total_seconds": float(np.median([row["total_seconds"] for row in steps])),
        "final_model_sha256": model_hash(model),
    }


def warmup(config: dict) -> float:
    torch.set_flush_denormal(False)
    torch.manual_seed(619_999)
    model = make_model(config).train()
    value = direction(255)
    batch = batch_for(value)[:8]
    started = time.perf_counter()
    loss = model(batch)
    loss.backward()
    return time.perf_counter() - started


def cpu_metadata() -> dict:
    return {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "affinity_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "gpu_used": False,
    }


def main() -> None:
    started = time.perf_counter()
    config = json.loads(CONFIG.read_text())
    largest = direction(0)
    smallest = direction(255)
    warmup_seconds = warmup(config)
    conditions = {}
    order = [
        ("smallest_default", smallest, False),
        ("largest_default", largest, False),
        ("largest_flush", largest, True),
        ("smallest_flush", smallest, True),
    ]
    for label, value, flush in order:
        conditions[label] = run_condition(label, value, flush, config)
    torch.set_flush_denormal(False)

    control = {
        "control": "twice-scaled direction asserted unit norm",
        "expected_to_fail": True,
        "observed_norm": float(torch.linalg.norm(2 * largest)),
        "rejected": abs(float(torch.linalg.norm(2 * largest)) - 1.0) > 1e-6,
    }
    result = {
        "scientific_status": "CPU_RUNTIME_DIAGNOSTIC_ONLY; Claims 1 and 4 remain BLOCKED",
        "order": [label for label, _, _ in order],
        "seeds": {
            "model": 620_001,
            "data": 620_002,
            "dsm_steps": [620_101, 620_102],
        },
        "warmup_seconds": warmup_seconds,
        "conditions": conditions,
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
        "largest_default_median_seconds": conditions["largest_default"][
            "median_total_seconds"
        ],
        "largest_flush_median_seconds": conditions["largest_flush"][
            "median_total_seconds"
        ],
        "smallest_default_median_seconds": conditions["smallest_default"][
            "median_total_seconds"
        ],
        "smallest_flush_median_seconds": conditions["smallest_flush"][
            "median_total_seconds"
        ],
        "largest_speedup": independent.get("largest_speedup"),
        "largest_loss_relative_error": independent.get(
            "largest_loss_relative_error"
        ),
        "mechanism_supported": independent.get("mechanism_supported"),
        "independent_checker_passed": independent.get("passed"),
        "total_runtime_seconds": result["total_runtime_seconds"],
        "scientific_status": result["scientific_status"],
    }
    (ARTIFACT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_CPU_DIAGNOSTIC_SUMMARY=" + json.dumps(summary, sort_keys=True))
    print("CLAIM_1_4_CPU_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))
    passed = checker.returncode == 0 and control["rejected"]
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
