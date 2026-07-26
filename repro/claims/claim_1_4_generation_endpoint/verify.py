"""One exact, predeclared iDDPM endpoint run for Claims 1 and 4.

The two endpoint sibling nodes use the same code and seeds. Only design.json's
geometry index differs. A single node is partial evidence, never a claim verdict.
"""

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
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.modules import Diffuser


HERE = Path(__file__).resolve().parent
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1_4_generation_endpoint"
GEOMETRY_RAW = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_1_4_geometry_adaptive"
    / "raw_geometry.json"
)
CONFIG = ROOT / "configs" / "1x16x16-iddpm.json"
ARTIFACT.mkdir(parents=True, exist_ok=True)
RESOURCE_GATE_STEP = 10
MAX_SECONDS_PER_UPDATE = 12.0


def emit(message: str) -> None:
    print(message, flush=True)


def cpu_metadata() -> dict:
    affinity = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "affinity_cpus": affinity,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "gpu_used": False,
    }


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


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_direction(design: dict) -> tuple[torch.Tensor, dict]:
    geometry = json.loads(GEOMETRY_RAW.read_text())
    index = str(design["geometry_index"])
    direction = torch.tensor(
        geometry["selected_eigenvectors"][index], dtype=torch.float32
    ).reshape(1, 16, 16)
    direction /= torch.linalg.norm(direction)
    return direction, {
        "geometry_index": design["geometry_index"],
        "endpoint": design["endpoint"],
        "eigenvalue": geometry["selected_eigenvalues"][index],
        "norm": float(torch.linalg.norm(direction)),
        "sha256": tensor_sha256(direction),
        "geometry_source_status": geometry["status"],
        "geometry_source_ready": geometry["generation_direction_ready"],
    }


def make_rank_one_data(
    direction: torch.Tensor, count: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coefficients = torch.randn(count, 1, 1, 1, generator=generator)
    return math.sqrt(direction.numel()) * coefficients * direction


def train(
    model: Diffuser, direction: torch.Tensor, config: dict, design: dict
) -> tuple[dict, torch.Tensor]:
    data = make_rank_one_data(direction, 10_000, design["training_data_seed"])
    shuffle_generator = torch.Generator(device="cpu").manual_seed(
        design["shuffle_seed"]
    )
    loader = DataLoader(
        TensorDataset(data),
        batch_size=1_000,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=shuffle_generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    torch.manual_seed(design["training_noise_seed"])
    checkpoints = {1, 10, 100, 250, 500, 1_000, 1_500, 2_000}
    curve = []
    recent = []
    resource_gate = None
    step = 0
    started = time.perf_counter()
    model.train()
    for epoch in range(200):
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            step += 1
            recent.append(float(loss.detach()))
            if len(recent) > 100:
                recent.pop(0)
            if step in checkpoints:
                row = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "trailing_100_mean": float(np.mean(recent)),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                curve.append(row)
                emit("CLAIM_1_4_TRAIN_PROGRESS=" + json.dumps(row, sort_keys=True))
                if step == RESOURCE_GATE_STEP:
                    seconds_per_update = row["elapsed_seconds"] / step
                    resource_gate = {
                        "step": RESOURCE_GATE_STEP,
                        "seconds_per_update": seconds_per_update,
                        "maximum_seconds_per_update": MAX_SECONDS_PER_UPDATE,
                        "passed": seconds_per_update <= MAX_SECONDS_PER_UPDATE,
                        "selection_basis": (
                            "host throughput only; independent of loss, samples, "
                            "metric values, direction, and scientific outcome"
                        ),
                    }
                    emit(
                        "CLAIM_1_4_RESOURCE_GATE="
                        + json.dumps(resource_gate, sort_keys=True)
                    )
                    if not resource_gate["passed"]:
                        raise RuntimeError(
                            "RESOURCE_HOST_TOO_SLOW: "
                            f"{seconds_per_update:.6f} seconds/update exceeds "
                            f"the preregistered {MAX_SECONDS_PER_UPDATE:.6f} limit"
                        )
    if step != 2_000:
        raise RuntimeError(f"expected 2000 optimizer steps, observed {step}")
    if resource_gate is None or not resource_gate["passed"]:
        raise RuntimeError("resource throughput gate did not pass")
    coefficients = (data.reshape(10_000, -1) @ direction.reshape(-1)).numpy()
    empirical_trace = float(np.var(coefficients, ddof=0))
    return {
        "optimizer": "Adam",
        "learning_rate": config["learning_rate"],
        "grad_clip": config["grad_clip"],
        "epochs": 200,
        "batch_size": 1_000,
        "optimizer_steps": step,
        "loss_curve": curve,
        "runtime_seconds": time.perf_counter() - started,
        "training_data_empirical_covariance_trace": empirical_trace,
        "training_data_expected_covariance_trace": 256.0,
        "model_state_sha256": model_sha256(model),
        "resource_gate": resource_gate,
    }, data


def sample(model: Diffuser, design: dict) -> tuple[torch.Tensor, dict]:
    model.eval()
    torch.manual_seed(design["sampling_seed"])
    output = torch.empty(10_000, 1, 16, 16)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index in range(10):
            init = torch.randn(1_000, 1, 16, 16)
            generated = model.sample(init=init, steps=None, eta=1.0)
            if not torch.isfinite(generated).all():
                raise RuntimeError(f"non-finite generated batch {batch_index}")
            output[batch_index * 1_000 : (batch_index + 1) * 1_000] = generated
            emit(
                "CLAIM_1_4_SAMPLE_PROGRESS="
                + json.dumps(
                    {
                        "batch": batch_index + 1,
                        "of": 10,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                )
            )
    return output, {
        "generated_samples": 10_000,
        "batch_size": 1_000,
        "batches": 10,
        "diffusion_steps": 1_000,
        "eta": 1.0,
        "runtime_seconds": time.perf_counter() - started,
        "samples_sha256": tensor_sha256(output),
    }


def metric(
    target: torch.Tensor,
    generated: torch.Tensor,
    direction: torch.Tensor,
    projection_seed: int,
) -> tuple[dict, dict]:
    target_flat = target.reshape(10_000, 256)
    generated_flat = generated.reshape(10_000, 256)
    generator = torch.Generator(device="cpu").manual_seed(projection_seed)
    per_projection = []
    started = time.perf_counter()
    projection_count = 16_384
    chunk_size = 256
    with torch.inference_mode():
        for start in range(0, projection_count, chunk_size):
            count = min(chunk_size, projection_count - start)
            projections = torch.randn(count, 256, generator=generator)
            projections /= torch.linalg.norm(projections, dim=1, keepdim=True)
            target_proj = torch.sort(target_flat @ projections.T, dim=0).values
            generated_proj = torch.sort(
                generated_flat @ projections.T, dim=0
            ).values
            values = torch.mean((target_proj - generated_proj) ** 2, dim=0)
            per_projection.extend(values.to(torch.float64).tolist())
            if (start // chunk_size + 1) % 8 == 0:
                emit(
                    "CLAIM_1_4_METRIC_PROGRESS="
                    + json.dumps(
                        {
                            "projections_complete": start + count,
                            "of": projection_count,
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    )
                )
    values = np.asarray(per_projection, dtype=np.float64)
    along = generated_flat @ direction.reshape(-1)
    total_sq = torch.sum(generated_flat**2, dim=1)
    orthogonal_sq = torch.clamp(total_sq - along**2, min=0)
    quantile_levels = torch.linspace(0, 1, 21)
    diagnostics = {
        "generated_along_direction_mean": float(torch.mean(along)),
        "generated_along_direction_std": float(torch.std(along, correction=0)),
        "generated_orthogonal_energy_mean": float(torch.mean(orthogonal_sq)),
        "generated_total_energy_mean": float(torch.mean(total_sq)),
        "along_direction_quantiles_21": torch.quantile(
            along, quantile_levels
        ).tolist(),
        "orthogonal_energy_quantiles_21": torch.quantile(
            orthogonal_sq, quantile_levels
        ).tolist(),
    }
    return {
        "name": "Sliced Wasserstein-2 and maximum Sliced Wasserstein-2",
        "projection_distribution": "normalized Gaussian",
        "projection_count": projection_count,
        "projection_chunk_size": chunk_size,
        "sw2": float(np.sqrt(values.mean())),
        "msw2": float(np.sqrt(values.max())),
        "projection_mean_squared_w2": per_projection,
        "runtime_seconds": time.perf_counter() - started,
    }, diagnostics


def negative_control() -> dict:
    bad_direction_norm = 2.0
    bad_covariance_trace = 1.0
    failures = {
        "unit_direction": abs(bad_direction_norm - 1.0) > 1e-12,
        "D_covariance_scaling": abs(bad_covariance_trace - 256.0) > 1e-12,
    }
    return {
        "control": "non-unit direction with unscaled N(0,vv^T) data",
        "expected_to_fail": True,
        "bad_direction_norm": bad_direction_norm,
        "bad_covariance_trace": bad_covariance_trace,
        "failed_assumption_checks": failures,
        "rejected": all(failures.values()),
    }


def main() -> None:
    total_started = time.perf_counter()
    design = json.loads((HERE / "design.json").read_text())
    config = json.loads(CONFIG.read_text())
    direction, direction_record = load_direction(design)
    compute = cpu_metadata()
    emit("CLAIM_1_4_ENDPOINT_COMPUTE=" + json.dumps(compute, sort_keys=True))

    torch.manual_seed(design["model_seed"])
    model = make_model(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    training, training_data = train(model, direction, config, design)
    generated, sampling = sample(model, design)
    target = make_rank_one_data(direction, 10_000, design["metric_target_seed"])
    metric_record, diagnostics = metric(
        target, generated, direction, design["projection_seed"]
    )
    control = negative_control()

    setup = {
        "architecture": "author iDDPM U-Net",
        "config": "configs/1x16x16-iddpm.json",
        "shape": [1, 16, 16],
        "dimension": 256,
        "model_parameters": parameter_count,
        "training_samples": 10_000,
        "target_distribution": "N(0, 256 v v^T)",
        "fixed_command": "uv sync --locked && .venv/bin/python repro/run_campaign.py",
    }
    exact_setup_passed = (
        direction_record["geometry_source_ready"]
        and abs(direction_record["norm"] - 1.0) <= 1e-6
        and training["optimizer_steps"] == 2_000
        and sampling["generated_samples"] == 10_000
        and sampling["diffusion_steps"] == 1_000
        and metric_record["projection_count"] == 16_384
        and math.isfinite(metric_record["sw2"])
        and math.isfinite(metric_record["msw2"])
        and control["rejected"]
    )
    result = {
        "claims": [1, 4],
        "scientific_status": "PARTIAL_SINGLE_ENDPOINT_SINGLE_SEED",
        "claim_verdict": "BLOCKED_PENDING_PAIRED_MULTI_SEED_AGGREGATION",
        "design": design,
        "setup": setup,
        "direction": direction_record,
        "training": training,
        "sampling": sampling,
        "metric": metric_record,
        "sample_diagnostics": diagnostics,
        "negative_control": control,
        "compute": compute,
        "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "total_runtime_seconds": time.perf_counter() - total_started,
        "exact_setup_passed": exact_setup_passed,
    }
    raw_path = ARTIFACT / "raw_results.json"
    raw_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    checker = subprocess.run(
        [sys.executable, str(HERE / "independent_check.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    emit(checker.stdout.rstrip())
    checker_output = (
        json.loads((ARTIFACT / "independent_checker_output.json").read_text())
        if checker.returncode == 0
        else {"passed": False, "exit_code": checker.returncode}
    )
    summary = {
        "endpoint": design["endpoint"],
        "geometry_index": design["geometry_index"],
        "eigenvalue": direction_record["eigenvalue"],
        "sw2": metric_record["sw2"],
        "msw2": metric_record["msw2"],
        "optimizer_steps": training["optimizer_steps"],
        "generated_samples": sampling["generated_samples"],
        "projection_count": metric_record["projection_count"],
        "training_runtime_seconds": training["runtime_seconds"],
        "sampling_runtime_seconds": sampling["runtime_seconds"],
        "metric_runtime_seconds": metric_record["runtime_seconds"],
        "total_runtime_seconds": result["total_runtime_seconds"],
        "exact_setup_passed": exact_setup_passed,
        "independent_checker_passed": checker_output.get("passed", False),
        "scientific_status": result["scientific_status"],
    }
    (ARTIFACT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    emit("CLAIM_1_4_ENDPOINT_SUMMARY=" + json.dumps(summary, sort_keys=True))
    emit("CLAIM_1_4_ENDPOINT=" + json.dumps(result, sort_keys=True))
    passed = exact_setup_passed and checker.returncode == 0
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
