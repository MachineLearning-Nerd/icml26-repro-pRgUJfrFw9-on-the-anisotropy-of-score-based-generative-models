from pathlib import Path
from PIL import Image
import torch as th
from torch.utils.data import Dataset
import torchvision as tv
from typing import Optional, Union


def ImageDataset(root: str, *args, **kwargs) -> th.Tensor:
    """
    torchvision.datasets.ImageFolder with simple PIL convert.
    """

    def loader(path: Union[str, Path]) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert()

    dataset = tv.datasets.ImageFolder(*args,
                                      **kwargs,
                                      root=root,
                                      loader=loader,
                                      transform=tv.transforms.Compose([
                                          tv.transforms.ToTensor(),
                                          tv.transforms.Normalize(mean=[0.5], std=[0.5])
                                          ]))
    return th.stack([data[0] for data in dataset], dim=0)


# Datasets
# ------------------------------------------------------------------------------------------

class NormalVVTDataset(Dataset):

    def __init__(self,
                 v: th.Tensor,
                 num_samples: int = 10000) -> None:
        """
        ~Normal(0, Dvv^T) where v is unit norm.

        Args:
            v: the basis vector (no batch dim).
            num_samples: number of samples to generate.
        """

        super().__init__()
        self.v = v
        self.num_samples = num_samples
        self.data = th.numel(v) ** 0.5 * v * th.randn(num_samples, *(v.ndim * [1]))

    def __getitem__(self, index: int) -> th.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.num_samples


class NSphereDataset(Dataset):
    """
    Dataset of points on the unit hyper-sphere supported on vs, rest dims are empty.
    """

    def __init__(self, vs: th.Tensor, num_samples: int = 10000) -> None:
        """
        Args:
            vs: the basis vectors stacked along the last dimension.
            num_samples: number of samples to generate.
        """

        super().__init__()
        self.vs = vs
        self.num_samples = num_samples
        n = vs.shape[-1]
        ambient_dim = th.prod(th.tensor(vs.shape[:-1])).item()
        flatv = vs.reshape(-1, n)
        coeffs = th.randn(n, num_samples)
        coeffs /= coeffs.norm(dim=0, keepdim=True)
        self.data = flatv @ coeffs * ambient_dim ** 0.5
        self.data = self.data.T.reshape(-1, *vs.shape[:-1])

    def __getitem__(self, index: int) -> th.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.num_samples


class GeometryAdaptiveImageDataset(Dataset):
    """
    Dataset that adapts to the geometry of the network.
    """

    def __init__(self,
                 root: str,
                 network_geometry_path: Optional[str],
                 data_geometry_path: Optional[str],
                 ascending_data_geometry: bool):
        super().__init__()
        # small-scale experiments can fit in memory
        self._data = ImageDataset(root)
        self.shape = self._data[0].shape
        self.flatten_shape = th.prod(th.tensor(self.shape))
        if network_geometry_path is None or data_geometry_path is None:
            self.mat = th.eye(self.flatten_shape)
            return
        network_vecs = th.load(network_geometry_path)["vecs"].reshape(-1, self.flatten_shape).t()
        data_vecs = th.load(data_geometry_path)["vecs"].reshape(-1, self.flatten_shape).t()
        if ascending_data_geometry:
            data_vecs = data_vecs.flip(dims=(1,))
        self.mat = network_vecs @ data_vecs.T
        self._data = self.fwd(self._data)

    def fwd(self, x: th.Tensor) -> th.Tensor:
        x = x.reshape(-1, self.flatten_shape).t()
        x = self.mat @ x
        return x.reshape(*self.shape, -1).moveaxis(-1, 0)

    def inv(self, x: th.Tensor) -> th.Tensor:
        x = x.reshape(-1, self.flatten_shape).t()
        x = self.mat.T @ x
        return x.reshape(*self.shape, -1).moveaxis(-1, 0)

    def __getitem__(self, index: int) -> th.Tensor:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> th.Tensor:
        """
        Returns the data in the original space.
        """
        return self.inv(self._data)
