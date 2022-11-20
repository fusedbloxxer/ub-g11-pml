from numpy.random import default_rng
from tensorflow import Tensor
import tensorflow as tw
import pandas as ps
import numpy as ny


def impute_missing_values(sample: Tensor, n_size: int) -> Tensor:
    """Fill in the missing values (rows) by computing the ewm."""
    sample_new = ps.DataFrame(columns=range(
        sample.shape[1]), index=range(n_size), dtype=ny.float64)

    # Generate indices to copy associated values
    gen = default_rng()
    sample_new_i = ny.sort(gen.choice(n_size, sample.shape[0], replace=False))

    # Copy old values while keeping same order
    sample_new.iloc[sample_new_i] = ps.DataFrame(sample.numpy())

    # Use interpolation to fill gaps and copy values where impossible
    gaps_filled = sample_new.interpolate(axis='index').fillna(method='bfill')
    return tw.convert_to_tensor(gaps_filled)


def fill_missing_values(dataset: list[Tensor], n_size: int) -> Tensor:
    """Receive a list of tensors of different sizes, fill in the missing values using
      a strategy (ex. ewm) then return a tensor of tensors of the same size."""

    for i, _ in enumerate(dataset):
        if dataset[i].shape[0] == n_size:
            continue
        elif dataset[i].shape[0] < n_size:
            dataset[i] = impute_missing_values(dataset[i], n_size)
        else:
            dataset[i] = dataset[i][:n_size]

    return tw.convert_to_tensor(dataset)


if __name__ == '__main__':
    print('running this file')
