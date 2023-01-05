
from typing import Any, Tuple, Dict
import matplotlib.pyplot as pt
import numpy as ny
import torch
import abc


class HistoryProgress(abc.ABC):
    """Base class used to display to the user the evolution of a model
    during the training and validation phases."""
    def __init__(self):
        pass

    @abc.abstractmethod
    def show(self) -> None:
        """Display to the user various statistics."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def accuracy(self):
        """Display the last train and/or valid accuracy value."""
        raise NotImplementedError()


class TrainValidHistoryProgress(HistoryProgress):
    """Class used over SKLearn's algorithms to extract the current
    training/validation accuracy for the last epoch."""
    def __init__(self, train_accuracy, valid_accuracy = None):
        super().__init__()
        self.train_accuracy = train_accuracy
        self.valid_accuracy = valid_accuracy

    def show(self) -> None:
        """Display to the user various statistics."""
        print(f'Train Accuracy: {self.train_accuracy}')
        if self.valid_accuracy:
          print(f'Valid Accuracy: {self.valid_accuracy}')

    @property
    def accuracy(self):
        """Display the last train and/or valid accuracy value."""
        return self.train_accuracy, self.valid_accuracy


class NNHistoryProgress(HistoryProgress):
    """History class used over Tensorflow models to extract the evolution
    over the training and/or validation phases across all epochs. Displays
    one or two subplots respectively."""
    def __init__(self, history: Any):
        super().__init__()
        self.history = history
        self.has_valid = 'val_loss' in self.history

    def show(self) -> None:
        """Show a plot containing the loss and accuracy for training and/or validation."""
        f, ax = pt.subplots(1, 2)
        f.tight_layout()
        for i, metric in enumerate(('categorical_accuracy', 'loss')):
            ax[i].grid(True)
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel(metric.replace('_', ' ').capitalize())
            ax[i].plot(ny.arange(len(self.history[metric])), self.history[metric], label='train')
        if self.has_valid:
            ax[0].plot(ny.arange(len(self.history['val_loss'])), self.history['val_categorical_accuracy'], label='valid')
            ax[0].legend()
            ax[1].plot(ny.arange(len(self.history['val_loss'])), self.history['val_loss'], label='valid')
            ax[1].legend()
        pt.show()

    @property
    def accuracy(self):
        """Return the last train and/or valid accuracy value."""
        return self.history['categorical_accuracy'][-1], \
               self.history['val_categorical_accuracy'][-1] if self.has_valid else None


class TorchHistoryProgress(HistoryProgress):
    """History class used over custom PyTorch models to plot one or two
    subplots for training and validation loss and/or accuracy phases respectively
    over all epochs."""
    def __init__(self, history: Dict[str, torch.Tensor]):
        super().__init__()
        self.history = history

    def show(self) -> None:
        """Show a plot containing the loss and/or accuracy for training and/or validation."""
        if not self.has_accuracy and not self.has_loss:
            print('Nothing to print')
            return
        elif self.has_accuracy and self.has_loss:
            f, ax = pt.subplots(1, 2, figsize=(10, 5))
            f.tight_layout()
            for i, metric in enumerate(('accuracy', 'loss')):
                for mode in ('train', 'valid'):
                    if f'{mode}_{metric}' in self.history:
                        ax[i].grid(True)
                        ax[i].set_xlabel('Epoch')
                        ax[i].set_ylabel(metric.capitalize())
                        ax[i].plot(ny.arange(len(self.history[f'{mode}_{metric}'])), \
                                                 self.history[f'{mode}_{metric}'], \
                                                 label=mode)
                ax[i].legend()
            pt.show()
            return
        # Display a single metric
        metric = 'accuracy' if self.has_accuracy else 'loss'
        f, ax = pt.subplots(1, 1, figsize=(10, 5))
        for mode in ('train', 'valid'):
            if f'{mode}_{metric}' in self.history:
                ax.grid(True)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.plot(ny.arange(len(self.history[f'{mode}_{metric}'])), \
                                      self.history[f'{mode}_{metric}'], \
                                      label=mode)
        ax.legend()
        pt.show()

    @property
    def train_accuracy(self) -> float | None:
        return self.accuracy[0]

    @property
    def train_loss(self) -> float | None:
        return self.loss[0]

    @property
    def valid_accuracy(self) -> float | None:
        return self.accuracy[1]

    @property
    def valid_loss(self) -> float | None:
        return self.loss[1]

    @property
    def has_accuracy(self) -> bool:
        return self.train_accuracy is not None or self.valid_accuracy is not None

    @property
    def has_loss(self) -> bool:
        return self.train_loss is not None or self.valid_loss is not None

    @property
    def accuracy(self) -> Tuple[float | None, float | None]:
        """Return the last train and/or valid accuracy value."""
        return self.history['train_accuracy'][-1] if 'train_accuracy' in self.history else None, \
               self.history['valid_accuracy'][-1] if 'valid_accuracy' in self.history else None

    @property
    def loss(self) -> Tuple[float | None, float | None]:
        """Return the last train and/or valid loss value."""
        return self.history['train_loss'][-1] if 'train_loss' in self.history else None, \
               self.history['valid_loss'][-1] if 'valid_loss' in self.history else None

