"""Independent integrity checker for an exact iDDPM endpoint result.

This module does not import or call the primary verifier.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1_4_generation_endpoint"


def main() -> None:
    raw = json.loads((ARTIFACT / "raw_results.json").read_text())
    values = np.asarray(raw["metric"]["projection_mean_squared_w2"], dtype=np.float64)
    recomputed_sw2 = float(np.sqrt(values.mean()))
    recomputed_msw2 = float(np.sqrt(values.max()))

    checks = {
        "projection_count": values.size == 16_384,
        "projection_values_finite_nonnegative": bool(
            np.isfinite(values).all() and (values >= 0).all()
        ),
        "sw2_recomputed": abs(recomputed_sw2 - raw["metric"]["sw2"]) <= 2e-6,
        "msw2_recomputed": abs(recomputed_msw2 - raw["metric"]["msw2"]) <= 2e-6,
        "direction_unit_norm": abs(raw["direction"]["norm"] - 1.0) <= 1e-6,
        "paper_scale_geometry": raw["direction"]["geometry_network_count"]
        == 1_000_000,
        "geometry_structurally_ready": raw["direction"][
            "geometry_source_ready"
        ]
        is True,
        "direction_stability_disclosed": isinstance(
            raw["direction"]["geometry_direction_stable"], bool
        ),
        "optimizer_steps": raw["training"]["optimizer_steps"] == 2_000,
        "generated_samples": raw["sampling"]["generated_samples"] == 10_000,
        "reverse_steps": raw["sampling"]["diffusion_steps"] == 1_000,
        "dataset_dimension": raw["setup"]["dimension"] == 256,
        "dataset_samples": raw["setup"]["training_samples"] == 10_000,
        "negative_control_rejected": raw["negative_control"]["rejected"] is True,
        "no_gpu": raw["compute"]["gpu_used"] is False,
        "common_four_thread_policy": raw["compute"]["torch_intraop_threads"] == 4
        and raw["setup"]["torch_intraop_threads"] == 4,
    }
    passed = all(checks.values())
    output = {
        "implementation": "independent NumPy aggregate and contract audit; no primary-verifier imports",
        "checks": checks,
        "recomputed_sw2": recomputed_sw2,
        "recomputed_msw2": recomputed_msw2,
        "passed": passed,
    }
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_1_4_ENDPOINT_INDEPENDENT=" + json.dumps(output, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
