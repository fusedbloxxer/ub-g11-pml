from __future__ import annotations
import copy
import pathlib as pb
from typing import Tuple, List, Optional
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

    def preprocess(self, rounding: Optional[int]=3) -> Dataset:
        """Preprocess the training and testing sets. Operations applied:
            - round each coordinate to a digit specified in `rounding` param."""
        if rounding is not None:
            self.train_data = ny.round(self.train_data, decimals=rounding)
            self.test_data = ny.round(self.test_data, decimals=rounding)
        return self

    def fill_gaps(self, n_size: int = 150, min_limit: Optional[int] = 75) -> Dataset:
        """Fill in interpolated elements in-between existing ones."""
        self.train_data, deleted_i = Dataset.fill_missing_values(self.train_data, n_size, min_limit)
        self.test_data, _ = Dataset.fill_missing_values(self.test_data, n_size)
        self.train_labels = ny.array([l for i, l in enumerate(self.train_labels)
                                        if i not in deleted_i])
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

    def to_pandas(self) -> ps.DataFrame:
        """Concatenate all training data samples into a single table, in order
        to perform statistics in a more flexible manner."""
        records = []
        for i, (data, label) in enumerate(zip(self.train_data, self.train_labels)):
            # Extract the column individually and append label and identifier
            x, y, z = data[:, 0], data[:, 1], data[:, 1]
            target = ny.full_like(x, label, dtype=ny.int64)
            id = ny.full_like(x, i, dtype=ny.int64)

            # Aggregate the values
            sample = ps.DataFrame({
                'id': id,
                'x': x,
                'y': y,
                'z': z,
                'label': target,
            })

            # Save to previously seen records
            records.append(sample)
        return ps.concat(records, ignore_index=True)

    def from_pandas(self, data: ps.DataFrame) -> Tuple[List[ny.ndarray], List[int]]:
        """Given a DataFrame that contains all info (see to_pandas), restore it
        to the original form as an array of arrays (samples) and an array (labels)."""
        data_labels, data_records = [], []
        for i in ps.unique(data['id']):
            # Extract all rows for the test entry
            data_sample = data[data['id'] == i].reset_index()

            # Extract the values as a ny array and item
            data_records.append(ny.array(data_sample[['x', 'y', 'z']]))
            data_labels.append(data_sample['label'].iloc[0].item())
        return data_records, data_labels

    def remove_outliers(self, by: str, factor: float = +15e-1) -> Dataset:
        """Remove outliers for each feature (x, y, z) by calculating the IQR,
        and using a factor to eliminate them.

        Args:
            by (str): If 'global' then compute the IQR over the entire dataset,
            otherwise if 'class' then compute the IQR for each class.
        """
        # Transform the dataset to easily compute stats
        train_data = self.to_pandas()

        if by == 'global':
            # Compute the boundary used during the elimination process
            qu_1 = train_data.quantile(q=25e-2)[['x', 'y', 'z']]
            qu_3 = train_data.quantile(q=75e-2)[['x', 'y', 'z']]
            iqr = ny.abs(qu_3 - qu_1)
            iqr_l = qu_1 - iqr * factor
            iqr_u = qu_3 + iqr * factor

            # Remove the outliers using IQR
            no_outliers = train_data[
                (train_data['x'] >= iqr_l['x']) & (train_data['x'] <= iqr_u['x']) &
                (train_data['y'] >= iqr_l['y']) & (train_data['y'] <= iqr_u['y']) &
                (train_data['z'] >= iqr_l['z']) & (train_data['z'] <= iqr_u['z'])
            ]
        elif by == 'class':
            def boundary(subset: ps.DataFrame, factor = 1.5):
                """Compute IQRs for each subset of samples of the same class."""
                qu_1 = subset.quantile(q=25e-2)[['x', 'y', 'z']]
                qu_3 = subset.quantile(q=75e-2)[['x', 'y', 'z']]
                iqr = ny.abs(qu_3 - qu_1)
                iqr_l = qu_1 - iqr * factor
                iqr_u = qu_3 + iqr * factor
                return ps.DataFrame({
                    'x': ps.Series([iqr_l['x'], iqr_u['x']], index=['iqr_l', 'iqr_u']),
                    'y': ps.Series([iqr_l['y'], iqr_u['y']], index=['iqr_l', 'iqr_u']),
                    'z': ps.Series([iqr_l['z'], iqr_u['z']], index=['iqr_l', 'iqr_u']),
                })

            # Group entries by class
            thresholds = train_data \
                .groupby('label', group_keys=True) \
                .apply(boundary)

            # Remove outliers from each class
            subsets = []
            for i in range(1, 21):
                subsets.append(train_data[
                    (train_data['label'] == i) &
                    (train_data['x'] >= thresholds.loc[i].loc['iqr_l', 'x']) &
                    (train_data['x'] <= thresholds.loc[i].loc['iqr_u', 'x']) &
                    (train_data['y'] >= thresholds.loc[i].loc['iqr_l', 'y']) &
                    (train_data['y'] <= thresholds.loc[i].loc['iqr_u', 'y']) &
                    (train_data['z'] >= thresholds.loc[i].loc['iqr_l', 'z']) &
                    (train_data['z'] <= thresholds.loc[i].loc['iqr_u', 'z'])
                ])
            no_outliers = ps.concat(subsets)

        # Transform back to the initial representation
        self.train_data, self.train_labels = self.from_pandas(no_outliers)
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
    def fill_missing_values(dataset: list[ny.ndarray], n_size: int, min_limit: Optional[int] = None) -> ny.ndarray:
        """Receive a list of tensors of different sizes, fill in the missing values using
        a strategy (ex. ewm) then return a tensor of tensors of the same size.
        Args:
            n_size: Specifies the number of rows expected at the output for each sample
            in the dataset.
            min_limit: Represents a lower-limit threshold which is used to
            discard samples that have too many missing datapoints.
        Returns:
            An array containing the dataset with the added/reduced values and
            an array containing the indices for the entries that were deleted.
        """
        dataset = copy.deepcopy(dataset)

        mark_for_deletion = []
        for i, _ in enumerate(dataset):
            if dataset[i].shape[0] == n_size:
                continue
            elif min_limit is not None and dataset[i].shape[0] < min_limit:
                mark_for_deletion.append(i)
            elif dataset[i].shape[0] < n_size:
                dataset[i] = Dataset.impute_missing_values(dataset[i], n_size)
            else:
                dataset[i] = dataset[i][:n_size]

        # When an element is deleted from the list decr the i to accomodate
        # for the shifted values
        for s, i in enumerate(mark_for_deletion):
            del dataset[i - s]

        return ny.array(dataset, dtype=ny.float32), mark_for_deletion


if __name__ == '__main__':
    print('running this file')
