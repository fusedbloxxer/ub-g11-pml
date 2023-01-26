import torch
import typing
import numpy as ny


class SceneDataset(torch.utils.data.Dataset):
  def __init__(self, data: ny.ndarray, labels: ny.ndarray) -> None:
    super().__init__()

    # Internal params
    self._data = torch.tensor(data)
    self._labels = torch.tensor(labels)

  def __getitem__(self, index: slice) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    # Do partial slicing for greater manipulation flexibility
    return self._data[index, ...], self._labels[index, ...]

  def __len__(self) -> int:
    return self._labels.shape[0]


def preprocess(data: ny.ndarray, grayscale: bool = True, min_max: bool = True) -> ny.ndarray:
  if min_max:
    # Map images from predefined space [0, 255] to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())

  if grayscale:
    # Consider that each color channel has a different way that it's perceived
    grayscale_transform = ny.repeat([0.3, 0.59, 0.11], repeats=150 ** 2).reshape((1, 3, 150, 150))

    # Transform to gray images
    data = ny.sum(data * grayscale_transform, axis=1)[:, None, ...]

  # Return the preprocessed images
  return data

