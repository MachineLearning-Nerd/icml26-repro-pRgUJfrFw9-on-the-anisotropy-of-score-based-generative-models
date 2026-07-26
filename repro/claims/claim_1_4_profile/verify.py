"""CPU cost calibration for the exact author iDDPM path.

This checker deliberately emits no scientific verdict for Claims 1 or 4.
"""

from __future__ import annotations

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.modules import Diffuser


ARTIFACT = Path(".openresearch/artifacts/claim_1_4_profile")
ARTIFACT.mkdir(parents=True, exist_ok=True)


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
            "patch_size": 2
        },
    )


def timed(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - started


def geometry_profile():
    durations = []
    output_norms = []
    for sample in range(6):
        torch.manual_seed(80_000 + sample)
        started = time.perf_counter()
        model = make_model().eval()
        timestep = torch.tensor([[[[137 * sample % 1000]]]], dtype=torch.int64)
        with torch.inference_mode():
            output = model.epsilon(torch.zeros(1, 1, 16, 16), timestep)
            output = output / model.sqrt_one_minus_alphas_cumprod[
                timestep.squeeze()
            ]
        durations.append(time.perf_counter() - started)
        output_norms.append(float(torch.linalg.norm(output)))
        del output, model
        gc.collect()
    return {
        "fresh_networks": len(durations),
        "seconds": durations,
        "median_seconds_per_network": float(np.median(durations)),
        "output_norms": output_norms,
        "paper_one_million_network_hours": float(np.median(durations) * 1_000_000 / 3600),
    }


def training_and_sampler_profile():
    torch.manual_seed(81_000)
    model = make_model().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 1000
    direction = torch.ones(1, 16, 16)
    direction /= torch.linalg.norm(direction)
    scalar = torch.randn(batch_size, 1, 1, 1)
    batch = math.sqrt(256) * scalar * direction

    # A small warm-up exercises the same graph without making the warm-up the
    # reported batch-1000 measurement.
    warmup = batch[:8]
    optimizer.zero_grad(set_to_none=True)
    warmup_loss = model(warmup)
    warmup_loss.backward()
    optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    train_seconds = time.perf_counter() - started

    model.eval()
    init = torch.randn(batch_size, 1, 16, 16)
    timestep = torch.full((batch_size, 1, 1, 1), 999, dtype=torch.int64)
    with torch.inference_mode():
        _, forward_seconds = timed(lambda: model.epsilon(init, timestep))
        _, second_forward_seconds = timed(lambda: model.epsilon(init, timestep))
    sampling_forward_seconds = min(forward_seconds, second_forward_seconds)
    result = {
        "batch_size": batch_size,
        "training_loss": float(loss),
        "batch_1000_optimizer_step_seconds": train_seconds,
        "paper_2000_training_step_hours": train_seconds * 2000 / 3600,
        "batch_1000_epsilon_forward_seconds": sampling_forward_seconds,
        "paper_10000_samples_1000_steps_model_calls": 10_000,
        "paper_sampling_model_forward_hours": sampling_forward_seconds * 10_000 / 3600,
    }
    del batch, init, loss, model, optimizer, scalar, timestep, warmup, warmup_loss
    gc.collect()
    return result


def sw2_pilot():
    samples = 2_000
    dimension = 256
    projections = 1_024
    generator = torch.Generator().manual_seed(82_000)
    first = torch.randn(samples, dimension, generator=generator)
    second = torch.randn(samples, dimension, generator=generator)
    directions = torch.randn(projections, dimension, generator=generator)
    directions /= torch.linalg.norm(directions, dim=1, keepdim=True)
    started = time.perf_counter()
    first_projected = torch.sort(first @ directions.T, dim=0).values
    second_projected = torch.sort(second @ directions.T, dim=0).values
    sw2 = torch.mean((first_projected - second_projected) ** 2).sqrt()
    elapsed = time.perf_counter() - started
    return {
        "samples": samples,
        "dimension": dimension,
        "projections": projections,
        "seconds": elapsed,
        "sw2": float(sw2),
        "paper_projection_count": 16_384,
        "linear_work_extrapolation_seconds": elapsed
        * (10_000 / samples)
        * (16_384 / projections),
        "warning": "projection/sort scaling is not perfectly linear; this is planning evidence only",
    }


def main():
    started = time.perf_counter()
    geometry = geometry_profile()
    train_sample = training_and_sampler_profile()
    metric = sw2_pilot()
    result = {
        "profile_complete": True,
        "scientific_status": "CALIBRATION_ONLY; Claims 1 and 4 remain BLOCKED",
        "architecture": {
            "shape": [1, 16, 16],
            "dimension": 256,
            "base_channels": 32,
            "channel_mults_as_constructed": [1, 1, 1],
            "residual_depth": 1,
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "paper_training_steps": 2000,
            "paper_batch_size": 1000,
            "paper_dataset_samples": 10000,
            "paper_geometry_networks": 1000000,
        },
        "cpu": {
            "logical": os.cpu_count(),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "geometry": geometry,
        "training_and_sampling": train_sample,
        "sw2_pilot": metric,
        "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "profile_runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
    }
    (ARTIFACT / "raw_profile.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_PROFILE=" + json.dumps(result, sort_keys=True))
    print(
        "CLAIM_1_4_PROFILE_SUMMARY "
        f"geometry_hours_1m={geometry['paper_one_million_network_hours']:.3f} "
        f"training_hours_2k={train_sample['paper_2000_training_step_hours']:.3f} "
        f"sampling_forward_hours={train_sample['paper_sampling_model_forward_hours']:.3f} "
        f"peak_rss_mb={result['peak_rss_megabytes']:.1f}"
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
