import argparse
from dataclasses import asdict
from utils import EXPERIMENTS
import json
import os
import sys
import time
import torch as th
from typing import Any, Callable, Dict, Set, Tuple


def main(extra_argparse: Set[str],
         make_run_fn: Callable[[Dict[str, Any]], Tuple[Callable, str]]) -> None:
    """
    Entry point for all scripts.

    Args:
        extra_argparse: list of additional, experiment-specific arguments to parse.
        make_run_fn: takes the arguments and returns the training and evaluation loop fn + a setting string.

    Returns:
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1000)
    for arg in extra_argparse:
        parser.add_argument("--" + arg, type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--config_path", type=str, default="configs/1x16x16-iddpm.json")
    parser.add_argument("--interpolation", type=str, default="nearest")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--save_samples", action="store_true")
    # cleaner to leave these off of the extra_argparse, keep only ints
    parser.add_argument("--network_geometry_path", type=str, default=None)
    # arguments for real data
    parser.add_argument("--data_geometry_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--ascending_data_geometry", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    th.cuda.reset_peak_memory_stats()

    var_args = vars(args)
    repeat = var_args.pop("repeat")
    save_path = var_args.pop("save_path")
    save_samples = var_args.pop("save_samples")
    config_path = var_args.pop("config_path")
    interpolation = var_args.pop("interpolation")
    patch_size = var_args.pop("patch_size")

    with open(config_path, "r") as f:
        config = json.load(f)
    config["diffuser"]["model"]["downsample_with_pool"] = (interpolation == "nearest")
    config["diffuser"]["model"]["interpolation"] = interpolation
    config["diffuser"]["model"]["patch_size"] = patch_size

    var_args["shape"] = config["shape"]
    run_fn, setting = make_run_fn(var_args)
    var_args.pop("shape")
    var_args.pop("network_geometry_path")
    var_args.pop("data_geometry_path")
    var_args.pop("data_dir")
    var_args.pop("ascending_data_geometry")
    var_args.pop("num_samples")

    swds = []
    mswds = []
    samples = []
    for _ in range(repeat):
        swd, mswd, sample = run_fn(**var_args, config=config)
        swds.append(swd)
        mswds.append(mswd)
        if save_samples:
            samples.append(sample)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    th.save({"swds": swds, "mswds": mswds, "samples": samples}, save_path)

    swd = th.tensor(swds).median().item()
    mswd = th.tensor(mswds).median().item()

    print(f"{setting} swd: {swd}, mswd: {mswd} Time: {time.perf_counter() - start:.2f} s, Max Mem: {th.cuda.max_memory_allocated() / 1e9:.2f} GB")  # noqa: E501


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage: python main.py <exp_code>, where <exp_code> is one of: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)
    exp_code = sys.argv[1]
    del sys.argv[1]

    if exp_code not in EXPERIMENTS:
        print(f"Unknown experiment code: {exp_code}, available codes are: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)

    main(**asdict(EXPERIMENTS[exp_code]))
