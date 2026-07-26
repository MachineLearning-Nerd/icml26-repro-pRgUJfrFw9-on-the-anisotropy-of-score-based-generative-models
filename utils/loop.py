import math
from utils.modules import Diffuser
import torch as th
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, Optional, Tuple
from utils.datasets import GeometryAdaptiveImageDataset


# Sliced Wasserstein Distance
# ------------------------------------------------------------------------------------------

def calculate_sliced_wassersteinp(dist1: th.Tensor,
                                  dist2: th.Tensor,
                                  num_projections: Optional[int] = None,
                                  p: float = 2) -> Tuple[float, float]:
    """
    Calculate the sliced Wasserstein distance (SWD) and maximum sliced Wasserstein distance (MSWD).

    Args:
        dist1: first distribution (N, D).
        dist2: second distribution (N, D).
        num_projections: number of random projections to use. Defaults to 64xD.
        p: the p-norm to use for the distance calculation.

    Returns:
        swd: sliced Wasserstein distance.
        mswd: maximum sliced Wasserstein distance.
    """

    if num_projections is None:
        num_projections = 64 * dist1.shape[1]

    # can also be done in a loop to avoid memory issues but for small-scale experiments this is fine
    projections = th.randn(num_projections, dist1.shape[1], device=dist1.device)
    projections = projections / projections.norm(dim=1, keepdim=True)

    proj_dist1 = dist1 @ projections.T
    proj_dist2 = dist2 @ projections.T

    proj_dist1, _ = th.sort(proj_dist1, dim=0)
    proj_dist2, _ = th.sort(proj_dist2, dim=0)

    diff = (proj_dist1 - proj_dist2).abs() ** p
    swd = diff.mean() ** (1 / p)
    mswd = diff.mean(dim=0).pow(1 / p).max()

    return swd.item(), mswd.item()


# Training and Evaluation Loop
# ------------------------------------------------------------------------------------------

def run(batch_size: int,
        epochs: int,
        dataset_fn: Callable[[], Dataset],
        config: Dict[str, Any]) -> Tuple[float, float, th.Tensor]:
    """
    Run the training and evaluation loop.

    Args:
        batch_size: the batch size for training.
        epochs: number of training epochs.
        dataset_fn: a callable that returns the dataset.
        config: configuration dictionary.

    Returns:
        swd: sliced Wasserstein distance.
        mswd: maximum sliced Wasserstein distance.
        sample: generated samples.
    """

    device = "cuda" if th.cuda.is_available() else "cpu"

    model = Diffuser(shape=config["shape"],
                     T=config["diffuser"]["T"],
                     linear=config["diffuser"]["linear"],
                     model_cfg=config["diffuser"]["model"])

    dataset = dataset_fn()
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            shuffle=True,
                            num_workers=0)

    model.to(device)

    optim = th.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for _ in range(epochs):
        for x in dataloader:
            x = x.to(device)
            optim.zero_grad()

            loss = model(x)

            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optim.step()

    with th.inference_mode():

        model.eval()

        num_samples = len(dataset)

        num_batches = num_samples // batch_size
        synth_data = th.zeros([num_samples] + config["shape"], device=device)

        for num_batch in range(num_batches):
            samples = model.sample(init=th.randn([batch_size] + config["shape"]).to(device))
            synth_data[num_batch * batch_size:(num_batch + 1) * batch_size, ...] = samples

        synth_data = synth_data.cpu()
        if isinstance(dataset, GeometryAdaptiveImageDataset):
            synth_data = dataset.inv(synth_data)

        swd, mswd = calculate_sliced_wassersteinp(dataset_fn()
                                                  .data
                                                  .reshape(num_samples, -1),
                                                  synth_data.reshape(num_samples, -1))

        if not math.isfinite(swd):
            raise ValueError("Non-finite values.")
        return swd, mswd, synth_data
