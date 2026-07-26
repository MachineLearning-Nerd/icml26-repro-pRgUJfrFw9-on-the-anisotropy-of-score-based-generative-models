import json
import numpy as np
import os
import time
import torch as th
from typing import Optional, Tuple
from utils import Diffuser, ImageDataset


# TODO: can this be accelerated with vmap?
# storing all samples is expensive but for small-scale experiments this is fine

def sample_networks(num_samples: int,
                    config_path: str,
                    data_dir: Optional[str],
                    interpolation: str,
                    patch_size: int,
                    sigma: float) -> th.Tensor:

    device = "cuda" if th.cuda.is_available() else "cpu"

    with open(config_path, "r") as f:
        config = json.load(f)
    config["diffuser"]["model"]["downsample_with_pool"] = (interpolation == "nearest")
    config["diffuser"]["model"]["interpolation"] = interpolation
    config["diffuser"]["model"]["patch_size"] = patch_size

    samples = th.empty([num_samples] + config["shape"], device=device)
    data = ImageDataset(data_dir) if data_dir else None

    def probe_dist(model: Diffuser) -> Tuple[th.Tensor, th.Tensor]:
        timestep = model.randint(batch_size=1, device=device)
        if data is None:
            xt = th.randn([1] + config["shape"], device=device) * sigma
        else:
            xt = data[th.randint(0, len(data), (1,))].to(device)
            xt = model.noise(xt, timestep)
        return xt, timestep

    for i in range(num_samples):
        model = Diffuser(shape=config["shape"],
                         T=config["diffuser"]["T"],
                         linear=config["diffuser"]["linear"],
                         model_cfg=config["diffuser"]["model"]).eval().to(device)
        xt, timestep = probe_dist(model)
        samples[i] = model.epsilon(xt,
                                   timestep).squeeze(0) / model.sqrt_one_minus_alphas_cumprod[timestep.squeeze()]

    return samples


@th.inference_mode()
def main(num_samples: int,
         config_path: Optional[str],
         data_dir: Optional[str],
         save_path: str,
         interpolation: str,
         patch_size: int,
         sigma: float,
         seed: Optional[int]) -> None:

    th.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    assert config_path is not None or data_dir is not None, "Either config_path or data_dir must be provided"

    if seed is not None:
        th.manual_seed(seed)

    if config_path is None:
        samples = ImageDataset(data_dir)
    else:
        samples = sample_networks(num_samples, config_path, data_dir, interpolation, patch_size, sigma)

    num_samples, shape = samples.shape[0], samples.shape[1:]
    samples = samples.cpu().reshape(num_samples, -1)

    geometry = samples.t() @ samples / num_samples
    vals, vecs = np.linalg.eigh(geometry.numpy())
    vals, vecs = th.tensor(vals), th.tensor(vecs)
    indices = vals.argsort(descending=True)
    vals = vals[indices]
    vecs = vecs[:, indices].t().reshape(-1, *shape)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    th.save({"vecs": vecs, "vals": vals}, save_path)

    print(f"Time: {time.perf_counter() - start:.2f} s, Max Mem: {th.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1000000)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--interpolation", type=str, default="nearest")
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    main(**vars(args))
