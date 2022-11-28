from __future__ import annotations
import copy
import pathlib as pb
from typing import Tuple
import pandas as ps
import numpy as ny
from numpy.random import default_rng

class Dataset(object):
    def __init__(self, path: pb.Path):
        # Load the dataset "as is" without applying preprocessing
        self.train_data,   \
        self.train_labels, \
        self.test_data = Dataset.load_dataset(path)

    def reshape(self, shape: Tuple[int, ...]) -> Dataset:
        """Reshape the training & testing data."""
        self.train_data = self.train_data.reshape(shape)
        self.test_data = self.test_data.reshape(shape)
        return self

    def fill_gaps(self) -> Dataset:
        """Fill in interpolated elements in-between existing ones."""
        self.train_data = Dataset.fill_missing_values(self.train_data, 150)
        self.test_data = Dataset.fill_missing_values(self.test_data, 150)
        self.train_labels = ny.array(self.train_labels)
        return self

    def normalize(self, strategy: str, axis: Tuple[int, ...], norm: int = 2, eps=1e-10) -> Dataset:
        """Normalizes the training and testing data using a custom strategy, over
        the specified dimensions. Assumes that the data has fixed size.
        Args:
            strategy:
            - standard
            - min_max
            - normalize

            axis: Tuple indicating the dimensions to compute over.
            norm: Indicates the order (rank) for the normalization operation
            eps: A small number added to avoid division by zero errors
        Returns: The current dataset with the train_data and test_data normalized internally.
        """
        if strategy == 'standard':
            m_train = self.train_data.mean(axis)
            s_train = self.train_data.std(axis)
            self.train_data = (self.train_data - m_train) / s_train
            self.test_data = (self.test_data - m_train) / s_train
        elif strategy == 'min_max':
            mi_train = self.train_data.min(axis)
            mx_train = self.train_data.max(axis)
            self.train_data = (self.train_data - mi_train) / (mx_train - mi_train + eps)
            mi_test = self.test_data.min(axis)
            mx_test = self.test_data.max(axis)
            self.test_data = (self.test_data - mi_test) / (mx_test - mi_test + eps)
        elif strategy == 'normalize':
            norm_train = ny.linalg.norm(self.train_data, ord=norm, axis=axis)
            norm_train = ny.expand_dims(norm_train, axis=axis)
            self.train_data /= norm_train
            norm_test = ny.linalg.norm(self.test_data, ord=norm, axis=axis)
            norm_test = ny.expand_dims(norm_test, axis=axis)
            self.test_data /= norm_test
        elif strategy is None:
            return self
        else:
            raise Exception(f'Invalid normalize strategy argument: {strategy}')
        return self

    @staticmethod
    def load_dataset(dataset_path: pb.Path) -> Tuple[list[ny.ndarray], list[ny.int64], list[ny.ndarray]]:
        """Read the whole dataset in memory available from the base path:
        Args:
            path:
                path/train/*.csv
                path/train_labels.csv
                path/test/*.csv
        Returns:
            A tuple containing three python lists:
            - train_data
            - train_labels
            - test_data
        """
        # Read train labels raw and use them for lookup
        train_labels_lookup = ps.read_csv(dataset_path / 'train_labels.csv')

        # Fetch eatch train sample and do lookup for the corresponding label
        train_data = []
        train_labels = []
        for sample_path in sorted((dataset_path / 'train').glob('*.csv')):
            # Lookup the label
            train_label = train_labels_lookup[train_labels_lookup['id'] == int(
                sample_path.stem)]
            train_labels.append(train_label['class'].item())

            # Read the sample and convert it to a tf compatible format
            train_entry = ps.read_csv(
                sample_path, header=None, names=['x', 'y', 'z'])
            train_data.append(train_entry.to_numpy())
        print('Number of training samples: ', len(train_data))
        print('Shape of a training sample: ', train_data[0].shape)

        # Read test samples
        test_data = []
        for sample_path in sorted((dataset_path / 'test').glob('*.csv')):
            test_entry = ps.read_csv(
                sample_path, header=None, names=['x', 'y', 'z'])
            test_data.append(test_entry.to_numpy())
        print('Number of testing samples: ', len(test_data))
        print('Shape of a testing sample: ', test_data[0].shape)
        return train_data, train_labels, test_data

    @staticmethod
    def impute_missing_values(sample: ny.ndarray, n_size: int) -> ny.ndarray:
        """Fill in the missing values (rows) by computing the ewm."""
        sample_new = ps.DataFrame(columns=range(sample.shape[1]),
                                  index=range(n_size),
                                  dtype=ny.float64)

        # Generate indices to copy associated values
        gen = default_rng()

        # To avoid filling the start and end of a sequence with repeated copies
        # of the same element, lock in the head and the tail and allow intermediary
        # interpolated values to occurr.
        sample_new_i = ny.sort(gen.choice(n_size - 2, sample.shape[0] - 2, replace=False) + 1)

        # Lock in the head and the tail
        sample_new.iloc[0], sample_new.iloc[-1] = sample[0], sample[-1]

        # Copy old values while keeping same order
        sample_new.iloc[sample_new_i] = ps.DataFrame(sample[1:-1])

        # Use interpolation to fill gaps and copy values where impossible
        gaps_filled = sample_new.interpolate(axis='index').fillna(method='bfill')
        return gaps_filled.to_numpy().astype(ny.float32)

    @staticmethod
    def fill_missing_values(dataset: list[ny.ndarray], n_size: int) -> ny.ndarray:
        """Receive a list of tensors of different sizes, fill in the missing values using
        a strategy (ex. ewm) then return a tensor of tensors of the same size."""
        dataset = copy.deepcopy(dataset)

        for i, _ in enumerate(dataset):
            if dataset[i].shape[0] == n_size:
                continue
            elif dataset[i].shape[0] < n_size:
                dataset[i] = Dataset.impute_missing_values(dataset[i], n_size)
            else:
                dataset[i] = dataset[i][:n_size]

        return ny.array(dataset, dtype=ny.float32)


if __name__ == '__main__':
    print('running this file')
