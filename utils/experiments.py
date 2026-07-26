import utils.bases as bases
from dataclasses import dataclass
from typing import Any, Callable, Dict, Set, Tuple
import torch as th
from torch.utils.data import Dataset
import utils.datasets as datasets
from utils.loop import run


@dataclass(frozen=True)
class ExperimentConfig:
    extra_argparse: Set[str]
    make_run_fn: Callable[[Dict[str, Any]], Tuple[Callable, str]]


def make_run_fn(dataset_fn: Callable[[], Dataset]):
    def run_fn(*args, **kwargs): return run(*args, **kwargs, dataset_fn=dataset_fn)
    return run_fn


def make_eigen_run_fn(var_args: Dict[str, Any]) -> Tuple[Callable, str]:
    idx = var_args.pop("idx")
    network_geometry_path = var_args["network_geometry_path"]
    num_samples = var_args["num_samples"]
    if network_geometry_path is None:
        raise ValueError("network_geometry_path must be provided")
    v = th.load(network_geometry_path)["vecs"][idx]
    def dataset_fn(): return datasets.NormalVVTDataset(v, num_samples=num_samples)
    setting = f"{idx}"
    return make_run_fn(dataset_fn), setting


def make_sphere_run_fn(var_args: Dict[str, Any]) -> Tuple[Callable, str]:
    idx1 = var_args.pop("idx1")
    idx2 = var_args.pop("idx2")
    idx3 = var_args.pop("idx3")
    network_geometry_path = var_args["network_geometry_path"]
    num_samples = var_args["num_samples"]
    if network_geometry_path is None:
        raise ValueError("network_geometry_path must be provided")
    vecs = th.load(network_geometry_path)["vecs"]
    v1, v2, v3 = vecs[idx1], vecs[idx2], vecs[idx3]
    vs = th.stack([v1, v2, v3], dim=-1)
    def dataset_fn(): return datasets.NSphereDataset(vs, num_samples=num_samples)
    setting = f"({idx1}, {idx2}, {idx3})"
    return make_run_fn(dataset_fn), setting


def make_dct_run_fn(var_args: Dict[str, Any]) -> Tuple[Callable, str]:
    fx = var_args.pop("fx")
    fy = var_args.pop("fy")
    num_samples = var_args["num_samples"]
    v = bases.dct_basis(var_args["shape"], fx=fx, fy=fy)
    def dataset_fn(): return datasets.NormalVVTDataset(v, num_samples=num_samples)
    setting = f"({fx}, {fy})"
    return make_run_fn(dataset_fn), setting


def make_dst_run_fn(var_args: Dict[str, Any]) -> Tuple[Callable, str]:
    fx = var_args.pop("fx")
    fy = var_args.pop("fy")
    num_samples = var_args["num_samples"]
    v = bases.dst_basis(var_args["shape"], fx=fx, fy=fy)
    def dataset_fn(): return datasets.NormalVVTDataset(v, num_samples=num_samples)
    setting = f"({fx}, {fy})"
    return make_run_fn(dataset_fn), setting


def make_hadamard_run_fn(var_args: Dict[str, Any]) -> Tuple[Callable, str]:
    sx = var_args.pop("sx")
    sy = var_args.pop("sy")
    num_samples = var_args["num_samples"]
    v = bases.hadamard_basis(var_args["shape"], sx=sx, sy=sy)
    def dataset_fn(): return datasets.NormalVVTDataset(v, num_samples=num_samples)
    setting = f"({sx}, {sy})"
    return make_run_fn(dataset_fn), setting


def make_haar_run_fn(var_args: Dict[str, Any]) -> Tuple[Callable, str]:
    scale = var_args.pop("scale")
    channel = var_args.pop("channel")
    x = var_args.pop("x")
    y = var_args.pop("y")
    num_samples = var_args["num_samples"]
    v = bases.haar_basis(var_args["shape"], scale, channel, x, y)
    def dataset_fn(): return datasets.NormalVVTDataset(v, num_samples=num_samples)
    setting = f"({scale}, {channel}, {x}, {y})"
    return make_run_fn(dataset_fn), setting


def make_canonical_run_fn(var_args: Dict[str, Any]) -> Tuple[Callable, str]:
    x = var_args.pop("x")
    y = var_args.pop("y")
    num_samples = var_args["num_samples"]
    v = bases.canonical_basis(var_args["shape"], x=x, y=y)
    def dataset_fn(): return datasets.NormalVVTDataset(v, num_samples=num_samples)
    setting = f"({x}, {y})"
    return make_run_fn(dataset_fn), setting


def make_images_run_fn(var_args: Dict[str, Any]) -> Tuple[Callable, str]:
    network_geometry_path = var_args["network_geometry_path"]
    data_geometry_path = var_args["data_geometry_path"]
    data_dir = var_args["data_dir"]
    ascending_data_geometry = var_args["ascending_data_geometry"]

    if data_dir is None:
        raise ValueError("data_dir must be provided")

    dataset = datasets.GeometryAdaptiveImageDataset(
        network_geometry_path=network_geometry_path,
        data_geometry_path=data_geometry_path,
        ascending_data_geometry=ascending_data_geometry,
        root=data_dir
    )
    def dataset_fn(): return dataset
    return make_run_fn(dataset_fn), f"{data_dir}"


EXPERIMENTS = {
    "eigen": ExperimentConfig(
        extra_argparse={"idx"},
        make_run_fn=make_eigen_run_fn
    ),
    "sphere": ExperimentConfig(
        extra_argparse={"idx1", "idx2", "idx3"},
        make_run_fn=make_sphere_run_fn
    ),
    "dct": ExperimentConfig(
        extra_argparse={"fx", "fy"},
        make_run_fn=make_dct_run_fn
    ),
    "dst": ExperimentConfig(
        extra_argparse={"fx", "fy"},
        make_run_fn=make_dst_run_fn
    ),
    "hadamard": ExperimentConfig(
        extra_argparse={"sx", "sy"},
        make_run_fn=make_hadamard_run_fn
    ),
    "haar": ExperimentConfig(
        extra_argparse={"scale", "channel", "x", "y"},
        make_run_fn=make_haar_run_fn
    ),
    "canonical": ExperimentConfig(
        extra_argparse={"x", "y"},
        make_run_fn=make_canonical_run_fn
    ),
    "images": ExperimentConfig(
        extra_argparse=set(),
        make_run_fn=make_images_run_fn
    ),
}
