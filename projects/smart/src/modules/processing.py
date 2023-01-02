from __future__ import annotations
import copy
import pathlib as pb
from typing import Tuple, List, Optional, Dict
import typing
import pandas as ps
import numpy as ny
from numpy.random import default_rng
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch


def preprocess(train_data_subset: List[ny.ndarray],
               train_labels_subset: List[int],
               valid_data_subset: ny.ndarray,
               valid_labels_subset: Optional[List[int]] = None,
               seed: int = 67) -> Tuple[ny.ndarray, ny.ndarray, ny.ndarray] \
                                | Tuple[ny.ndarray, ny.ndarray, ny.ndarray, ny.ndarray]:
    """Preprocess the data"""
    if valid_labels_subset is not None:
        # 1. Remove outliers
        # train_data_subset, \
        # train_labels_subset = remove_outliers(train_data_subset, train_labels_subset,
        #                                       by='global', factor=1.5)
        # valid_data_subset, \
        # valid_labels_subset = remove_outliers(valid_data_subset, valid_labels_subset,
        #                                       by='global', factor=1.5)

        # 2. Apply upper-bound, remove outliers, apply lower-bound, interpolate
        train_data_subset, \
        train_labels_subset = fill_gaps(train_data_subset, train_labels_subset, n_size=150)
        valid_data_subset, \
        valid_labels_subset = fill_gaps(valid_data_subset, valid_labels_subset, n_size=150)

        # 3. Normalize the data
        train_data_subset, \
        valid_data_subset = normalize(train_data_subset,
                                      valid_data_subset,
                                      strategy='standard', axis=(0,))

        # 4. Flatten the dimensions
        train_data_subset = train_data_subset.reshape((-1, 450))
        valid_data_subset = valid_data_subset.reshape((-1, 450))

        # 6. Reshuffle the train data because outliers orders asc by class
        shuffled_indices = ny.arange(len(train_data_subset))
        default_rng(seed).shuffle(shuffled_indices)
        train_data_subset = train_data_subset[shuffled_indices]
        train_labels_subset = train_labels_subset[shuffled_indices]

        # Return the preprocessed data
        return train_data_subset, train_labels_subset, \
               valid_data_subset, valid_labels_subset
    else:
        # 1. Remove outliers
        # train_data_subset, \
        # train_labels_subset = remove_outliers(train_data_subset,   \
        #                                       train_labels_subset, \
        #                                       by='global', factor=1.5)
        # valid_data_subset = remove_outliers(valid_data_subset,
        #                                     by='global', factor=1.5)

        # 2. Apply upper-bound, remove outliers, apply lower-bound, interpolate
        train_data_subset, \
        train_labels_subset = fill_gaps(train_data_subset, train_labels_subset, n_size=150)
        valid_data_subset = fill_gaps(valid_data_subset, n_size=150)

        # 3. Normalize the data
        train_data_subset, \
        valid_data_subset = normalize(train_data_subset, \
                                      valid_data_subset, \
                                      strategy='standard', axis=(0,))

        # 4. Flatten the dimensions
        train_data_subset = train_data_subset.reshape((-1, 450))
        valid_data_subset = valid_data_subset.reshape((-1, 450))

        # 6. Reshuffle the train data because outliers orders asc by class
        shuffled_indices = ny.arange(len(train_data_subset))
        default_rng(seed).shuffle(shuffled_indices)
        train_data_subset = train_data_subset[shuffled_indices]
        train_labels_subset = train_labels_subset[shuffled_indices]

        # Return the preprocessed data
        return train_data_subset, train_labels_subset, valid_data_subset


def normalize(train_data: ny.ndarray, test_data: ny.ndarray, strategy: Optional[str],
              axis: Tuple[int, ...], norm: int = 2, eps=1e-10) -> Tuple[ny.ndarray, ny.ndarray]:
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
        m_train = train_data.mean(axis)
        s_train = train_data.std(axis)
        train_data = (train_data - m_train) / (s_train + eps)
        test_data = (test_data - m_train) / (s_train + eps)
    elif strategy == 'min_max':
        mi_train = train_data.min(axis)
        mx_train = train_data.max(axis)
        train_data = (train_data - mi_train) / (mx_train - mi_train + eps)
        mi_test = test_data.min(axis)
        mx_test = test_data.max(axis)
        test_data = (test_data - mi_test) / (mx_test - mi_test + eps)
    elif strategy == 'normalize':
        norm_train = ny.linalg.norm(train_data, ord=norm, axis=axis)
        norm_train = ny.expand_dims(norm_train, axis=axis)
        train_data = train_data / (norm_train + eps)
        norm_test = ny.linalg.norm(test_data, ord=norm, axis=axis)
        norm_test = ny.expand_dims(norm_test, axis=axis)
        test_data = test_data / (norm_test + eps)
    elif strategy is None:
        return train_data, test_data
    else:
        raise Exception(f'Invalid "strategy" argument: {strategy}')
    return train_data, test_data


def to_pandas(dataset: typing.List[ny.ndarray], dataset_labels: Optional[List[int]] = None) -> ps.DataFrame:
    """Concatenate all data samples into a single table, in order to perform
    statistics in a more flexible manner."""
    records = []
    items = dataset if dataset_labels is None else zip(dataset, dataset_labels)
    for i, item in enumerate(items):
        # Extract the contained information
        data = item if dataset_labels is None else item[0]
        labels = None if dataset_labels is None else item[1]

        # Extract the column individually and append label and identifier
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        id = ny.full_like(x, i, dtype=ny.int64)

        # Aggregate the values
        sample = ps.DataFrame({
            'id': id,
            'x': x,
            'y': y,
            'z': z,
        })

        # Add labels as well if present
        if dataset_labels is not None:
            sample['label'] = ny.full_like(x, labels, dtype=ny.int64)

        # Save to previously seen records
        records.append(sample)
    return ps.concat(records, ignore_index=True)


def from_pandas(data: ps.DataFrame) -> Tuple[List[ny.ndarray], List[int]] | List[ny.ndarray]:
    """Given a DataFrame that contains all info (see to_pandas), restore it
    to the original form as an array of arrays (samples) and an array (labels?)."""
    data_labels = []
    data_records = []

    for i in ps.unique(data['id']):
        # Extract all rows for the test entry
        data_sample = data[data['id'] == i].reset_index()

        # Extract the values as a ny array and item
        data_records.append(ny.array(data_sample[['x', 'y', 'z']]))

        # Check for label presence first
        if 'label' in data_sample:
            data_labels.append(data_sample['label'].iloc[0].item())

    if 'label' in data:
        return data_records, data_labels
    else:
        return data_records


def remove_outliers(data: List[ny.ndarray], labels: Optional[List[int]] = None,
                    by: str = 'global', factor: float = +15e-1) -> Tuple[List[ny.ndarray], List[int]] | List[ny.ndarray]:
    """Remove outliers for each feature (x, y, z) by calculating the IQR,
    and using a factor to eliminate them.

    Args:
        by (str): If 'global' then compute the IQR over the entire dataset;
                  If 'class' then compute the IQR for each class;
                  If 'both' then use 'class' followed by 'global';
    """
    # Transform the dataset to easily compute stats
    dataset = to_pandas(data, labels)

    if labels is None and by == 'class':
        raise Exception('Cannot remove intraclass outliers without labels. Use global strategy.')

    if by == 'global':
        # Compute the boundary used during the elimination process
        qu_1 = dataset.quantile(q=25e-2)[['x', 'y', 'z']]
        qu_3 = dataset.quantile(q=75e-2)[['x', 'y', 'z']]
        iqr = ny.abs(qu_3 - qu_1)
        iqr_l = qu_1 - iqr * factor
        iqr_u = qu_3 + iqr * factor

        # Remove the outliers using IQR
        no_outliers = dataset[
            (dataset['x'] >= iqr_l['x']) & (dataset['x'] <= iqr_u['x']) &
            (dataset['y'] >= iqr_l['y']) & (dataset['y'] <= iqr_u['y']) &
            (dataset['z'] >= iqr_l['z']) & (dataset['z'] <= iqr_u['z'])
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
        thresholds = dataset \
            .groupby('label', group_keys=True) \
            .apply(boundary)

        # Remove outliers from each class
        subsets = []
        for i in range(1, 21):
            subsets.append(dataset[
                (dataset['label'] == i) &
                (dataset['x'] >= thresholds.loc[i].loc['iqr_l', 'x']) &
                (dataset['x'] <= thresholds.loc[i].loc['iqr_u', 'x']) &
                (dataset['y'] >= thresholds.loc[i].loc['iqr_l', 'y']) &
                (dataset['y'] <= thresholds.loc[i].loc['iqr_u', 'y']) &
                (dataset['z'] >= thresholds.loc[i].loc['iqr_l', 'z']) &
                (dataset['z'] <= thresholds.loc[i].loc['iqr_u', 'z'])
            ])
        no_outliers = ps.concat(subsets)
    elif by == 'both':
        if labels is None:
            raise Exception(f'Cannot use "both" strategy with no given labels.')
        data, labels = remove_outliers(data, labels, by='class', factor=factor)
        data, labels = remove_outliers(data, labels, by='global', factor=factor)
        return data, labels
    else:
        raise Exception(f'Invalid "by" argument value: {by}')

    return from_pandas(no_outliers)


def split_train_data(train_data: ny.ndarray, train_labels: ny.ndarray,
                     ratio: float = 8e-1, shuffle: bool = True, seed: int = 64) -> Tuple[ny.ndarray, ny.ndarray, ny.ndarray, ny.ndarray]:
    """Split the training dataset into validation and training according
    to the given ratio. Optionally apply shuffle to the data in order to
    obtain different outputs."""
    indices = ny.arange(train_data.shape[0])
    if shuffle:
        gen = default_rng(seed)
        indices = gen.choice(indices, size=indices.shape[0], replace=False)
    pivot_index = int(ny.floor(indices.shape[0] * ratio))

    # Index using the pivot and generated indices
    train_data = train_data[:pivot_index]
    valid_data = train_data[pivot_index:]
    train_labels = train_labels[:pivot_index]
    valid_labels = train_labels[pivot_index:]

    # Return split data partitions
    return train_data, train_labels, valid_data, valid_labels


def fill_gaps(data: List[ny.ndarray], labels: Optional[List[int]] = None, n_size: int = 150,
              min_limit: Optional[int] = None) -> Tuple[ny.ndarray, ny.ndarray] | ny.ndarray:
    """Fill in interpolated elements in-between existing ones."""
    filled_data, deleted_i = fill_missing_values(data, n_size, min_limit)

    # If labels are given then delete marked elements
    if labels is not None:
        filled_labels = ny.array([l for i, l in enumerate(labels)
                                    if i not in deleted_i])
        return filled_data, filled_labels

    # Otherwise return only the processed data
    return filled_data


def fill_missing_values(dataset: List[ny.ndarray], n_size: int, min_limit: Optional[int] = None) -> Tuple[ny.ndarray, List[int]]:
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
            dataset[i] = impute_missing_values(dataset[i], n_size)
        else:
            dataset[i] = dataset[i][:n_size]

    # When an element is deleted from the list decr the i to accomodate
    # for the shifted values
    for s, i in enumerate(mark_for_deletion):
        del dataset[i - s]

    return ny.array(dataset, dtype=ny.float32), mark_for_deletion


def impute_missing_values(sample: ny.ndarray, n_size: int, seed: int = 87) -> ny.ndarray:
    """Fill in the missing values (rows) by computing the ewm."""
    sample_new = ps.DataFrame(columns=range(sample.shape[1]),
                                index=range(n_size),
                                dtype=ny.float64)

    # Generate indices to copy associated values
    gen = default_rng(seed)

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


def load_dataset(dataset_path: pb.Path) -> Dict[str, Tuple[List[ny.ndarray], List[int]] | List[ny.ndarray]]:
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
    return {
        'train': (train_data, train_labels),
        'test': test_data
    }


class SmartDataset(data.Dataset):
    """Reshape the data from (N, F) to (N, E, 1, S), where
    N is the number of samples, E is the embedding size and S is the
    sequence length and subtract one from the labels: [1, 20] -> [0, 19].
    The transformed data is then exposed as (sample, label) pairs to the caller."""
    def __init__(self, data: ny.ndarray, labels: ny.ndarray | None = None,
        n_embed: int = 3, n_seq: int = 150) -> None:
        super().__init__()

        # Save given input data and reshape it as a 2D tensor with n_embed channels
        self.data_ = data.reshape((-1, 150, 3))
        self.data_ = ny.transpose(self.data_, axes=(0, 2, 1))
        self.data_ = ny.expand_dims(self.data_, axis=2)
        self.data_ = torch.tensor(self.data_)

        # Because the networks use class logits internally and argmax to
        # select the most probable class, we need to map [1, C] to [0, C) where
        # C represents the 20 classes.
        self.labels_ = None if labels is None else torch.tensor(labels) - 1

    @property
    def data(self) -> torch.Tensor:
        return self.data_

    @property
    def labels(self) -> torch.Tensor | None:
        return None if self.labels_ is None else self.labels_

    def __getitem__(self, index: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor | None]:
        # Select the desired sample and keep other dimensions to allow slicing
        x_sample = self.data_[index, ...]

        # Return both the sample and its label
        if self.labels_ is not None:
            return x_sample, self.labels_[index, ...]

        # Or only the sample if the label is missing
        return x_sample

    def __len__(self) -> int:
        return self.data_.shape[0]


def get_loaders(train_data: ny.ndarray, train_labels: ny.ndarray,
                valid_data: ny.ndarray = None, valid_labels: ny.ndarray = None,
                shuffle: bool = True, batch: int = 32, workers: int = 8,
                prefetch: int = 4) -> Tuple[data.DataLoader, data.DataLoader]:
    """Create and return data.DataLoader for the training data and validation if
    it's not None. The data is batched and the following operations are applied:
        - labels are mapped from [1, C] to [0, C - 1]
        - the data is reshaped from (N, F) to (N, C, T)

    Args:
        train_data (ny.ndarray): The training data to fit on later.
        train_labels (ny.ndarray): The training labels used for supervision.
        valid_data (ny.ndarray, optional): The validation data to evaluate the model. Defaults to None.
        valid_labels (ny.ndarray, optional): The validation labels to evaluate the model. Defaults to None.

    Returns:
        Tuple[data.DataLoader, data.DataLoader]: The training loader
        and the validation one if the provided data is not None, otherwise it's None.
    """
    train_loader = get_loader(train_data, train_labels, shuffle=shuffle, \
                              batch=batch, workers=workers, prefetch=prefetch)
    valid_loader = get_loader(valid_data, valid_labels, shuffle=shuffle, \
                              batch=batch, workers=workers, prefetch=prefetch) \
                    if valid_data is not None and valid_labels is not None else None
    return train_loader, valid_loader


def get_loader(data: ny.ndarray, labels: ny.ndarray | None = None, shuffle: bool = True,
               batch: int = 32, workers: int = 8, prefetch: int = 4) -> data.DataLoader:
    """Create and return a single dta.DataLoader. The data is batched and
    the following operations are applied:
        - labels are mapped from [1, C] to [0, C - 1]
        - the data is reshaped from (N, F) to (N, C, T)

    Args:
        data (ny.ndarray): The data to be provided to the model.
        labels (ny.ndarray): The labels used for supervision.

    Returns:
        data.DataLoader: A loader that gives back a mini-batch of paired input data and labels.
    """
    dataset = SmartDataset(data, labels)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=True,
                      prefetch_factor=prefetch)

