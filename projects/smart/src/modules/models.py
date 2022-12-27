from __future__ import annotations
from typing import TypedDict, TypeVar, Callable, Generic, List, Any, Tuple, Type, Iterator
import abc
import pathlib as pb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from nvitop.callbacks.keras import GpuStatsLogger
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as pt
import numpy as ny
import sklearn as sn
import tensorflow as tw
from .blocks import TemporalAttentionNN, SmartDataset
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import optimizer
import torch


class HyperParams(TypedDict):
    pass


class TrainParams(TypedDict):
    n_folds: int


class TCNNParams(HyperParams):
    optim: Callable[[], tw.keras.optimizers.Optimizer]
    m_init: tw.keras.initializers.VarianceScaling
    activ_fn: tw.keras.layers.Activation
    lr_update: Tuple[int, int, float]
    n_filters: int
    s_filters: int
    dropout: float
    n_units: int
    n_batch: int
    n_epochs: int


class AttentionTCNNParams(HyperParams): # TODO: add custom init
    optim: Callable[[Iterator[torch.nn.Parameter]], optimizer.Optimizer]
    sch_lr: Callable[[optimizer.Optimizer], torch.optim.lr_scheduler._LRScheduler]
    init_fn: Callable[[torch.Tensor], None]
    activ_fn: Type[nn.Module]
    bottleneck: int
    dropout: float
    n_filters: int
    s_filters: int
    n_epochs: int
    n_units: int
    n_batch: int
    norm: bool
    bias: bool


class SVMParams(HyperParams):
    C: float
    kernel: str
    gamma: str


class KNNParams(HyperParams):
    n_neighbors: int
    p: int


class BoostedTreesParams(HyperParams):
    learning_rate: float
    n_estimators: int
    subsample: float
    max_depth: int


class HistoryProgress(abc.ABC):
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
    def __init__(self, history: Tuple[torch.Tensor, ...]):
        super().__init__()

        self.history = {
            'train_loss': history[0],
            'train_accuracy': history[1],
        }

        if len(history) >= 3:
            self.history['valid_accuracy'] = history[2]
        if len(history) >= 4:
            self.history['valid_loss'] = history[3]

    def show(self) -> None:
        """Show a plot containing the loss and accuracy for training and/or validation."""
        f, ax = pt.subplots(1, 2, figsize=(10, 5))
        f.tight_layout()
        for i, metric in enumerate(('train_accuracy', 'train_loss')):
            ax[i].grid(True)
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel(metric.split('_')[1].capitalize())
            ax[i].plot(ny.arange(len(self.history[metric])), self.history[metric], label='train')
        for i, metric in enumerate(('valid_accuracy', 'valid_loss')):
            if not (metric in self.history):
                continue
            ax[i].plot(ny.arange(len(self.history[metric])), self.history[metric], label='valid')
            ax[i].legend()
        pt.show()

    @property
    def accuracy(self):
        """Return the last train and/or valid accuracy value."""
        return self.history['train_accuracy'][-1], \
               self.history['valid_accuracy'][-1] if 'valid_accuracy' in self.history else None

    @property
    def loss(self):
        """Return the last train and/or valid loss value."""
        return self.history['train_loss'][-1], \
               self.history['valid_loss'][-1] if 'valid_loss' in self.history else None


HP = TypeVar('HP', bound=HyperParams)


class Model(abc.ABC, Generic[HP]):
    hparams: HP

    def __init__(self, hparams: HP):
        super().__init__()
        self.hparams = hparams

    @abc.abstractmethod
    def fit(self, train_data: ny.ndarray, train_labels: ny.ndarray,
                  valid_data: ny.ndarray = None, valid_labels: ny.ndarray = None) -> HistoryProgress:
        """Train the model over the given data and labels and optionally predict over time,
        or at the end, on the validation set.
        Args:
            train_data: (n_samples, n_features)
            train_labels: (n_samples,)
            valid_data: (n_samples, n_features)
            valid_labels: (n_samples,)
        Returns: A history object containing statistics obtained throughout the training
        and validation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, data: ny.ndarray) -> ny.ndarray:
        """Predict an array of labels for the given data."""
        raise NotImplementedError()

    def accuracy(self, data: ny.ndarray, labels_t: ny.ndarray) -> float:
        """Compute the accuracy for the given data against the true labels."""
        labels_p = self.predict(data)
        return ny.mean(labels_p == labels_t)

    def conf(self, data: ny.ndarray, labels_t: ny.ndarray) -> ny.ndarray:
        """Compute the confusion matrix for the given data against the true labels."""
        labels_p = self.predict(data)
        return confusion_matrix(labels_t, labels_p)

TSKModel = TypeVar('TSKModel', bound=sn.base.ClassifierMixin |
                                     sn.base.BaseEstimator)

class SKModel(Model[HP], Generic[HP, TSKModel]):
    model_: TSKModel

    def __init__(self, hparams: HP, model: TSKModel):
        super().__init__(hparams=hparams)

        # Retrieve the model
        self.model_ = model

    def fit(self, train_data: ny.ndarray, train_labels: ny.ndarray,
                  valid_data: ny.ndarray = None, valid_labels: ny.ndarray = None) -> HistoryProgress:
        # Test if validation data was passed in
        has_valid = valid_data is not None and valid_labels is not None

        # Train the model on the available data
        self.model_.fit(train_data, train_labels)

        # Compute the accuracy over train & valid data
        t_acc = self.model_.score(train_data, train_labels)
        v_acc = self.model_.score(valid_data, valid_labels) if has_valid else None
        return TrainValidHistoryProgress(t_acc, v_acc)

    def predict(self, data: ny.ndarray) -> ny.ndarray:
        """Predict an array of labels for the given data."""
        return self.model_.predict(data)


class BoostedTreesModel(SKModel[BoostedTreesParams, GradientBoostingClassifier]):
    def __init__(self, hparams: BoostedTreesParams, verbose: int = 0):
        super().__init__(hparams, GradientBoostingClassifier(
            learning_rate=hparams['learning_rate'],
            n_estimators=hparams['n_estimators'],
            subsample=hparams['subsample'],
            max_depth=hparams['max_depth'],
            verbose=verbose,
        ))


class SVMModel(SKModel[SVMParams, SVC]):
    def __init__(self, hparams: SVMParams, verbose: bool = True):
        super().__init__(hparams=hparams, model=SVC(C=hparams['C'],
                                                    kernel=hparams['kernel'],
                                                    gamma=hparams['gamma'],
                                                    verbose=verbose))


class KNNModel(SKModel[KNNParams, KNeighborsClassifier]):
    def __init__(self, hparams: KNNParams):
        super().__init__(hparams=hparams, model=KNeighborsClassifier(
            n_neighbors=hparams['n_neighbors'],
            p=hparams['p'],
        ))


class TCNNModel(Model[TCNNParams]):
    def __init__(self, hparams: TCNNParams, root: pb.Path = None):
        super().__init__(hparams)

        # Instantiate the model
        self.model_ = self.__create_model()
        self.logs_dir_ = root / 'logs'
        self.n_workers = 8

    def fit(self, train_data: ny.ndarray, train_labels: ny.ndarray,
                  valid_data: ny.ndarray = None, valid_labels: ny.ndarray = None) -> HistoryProgress:
        # Reshape data to proper format
        if valid_data is not None and valid_labels is not None:
            valid_labels = ny.array(valid_labels) - 1 # [1, 20] -> [0, 19]
            valid_labels = tw.raw_ops.OneHot(indices=valid_labels, depth=20, on_value=1.0, off_value=0.0)
            validation_data = valid_data.reshape((-1, 1, 150, 3)), valid_labels
        else:
            validation_data = None
        train_data = train_data.reshape((-1, 1, 150, 3))

        # Map labels to [0, 19] and encode them using one-hot
        train_labels = ny.array(train_labels) - 1
        train_labels = tw.raw_ops.OneHot(indices=train_labels, depth=20, on_value=1.0, off_value=0.0)

        # Train the model and validate it on each step
        fit_history = self.model_.fit(train_data, train_labels,
                        validation_data=validation_data,
                        epochs=self.hparams['n_epochs'],
                        batch_size=self.hparams['n_batch'],
                        workers=self.n_workers, use_multiprocessing=True,
                        initial_epoch=0, callbacks=self.fitting_callbacks())

        # Sends stats to be plotted later
        return NNHistoryProgress(fit_history.history)

    def predict(self, data: ny.ndarray) -> ny.ndarray:
        """Predict an array of labels for the given data."""
        probs = self.model_.predict(data.reshape(-1, 1, 150, 3),
                            self.hparams['n_batch'],
                            workers=self.n_workers, use_multiprocessing=True)
        return ny.argmax(probs, axis=1) + 1 # [0, 19] -> [1, 20]

    def __create_model(self) -> tw.keras.Model:
        model_tcnn = tw.keras.Sequential([
          tw.keras.layers.InputLayer(input_shape=(1, 150, 3)),
          # --- First Convolutional Block ---
          tw.keras.layers.Conv2D(self.hparams['n_filters'],
                                 kernel_size=(1, self.hparams['s_filters']),
                                 padding='same',
                                 kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.Conv2D(self.hparams['n_filters'],
                                 kernel_size=(1, self.hparams['s_filters']),
                                 padding='same',
                                 kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
          tw.keras.layers.Dropout(self.hparams['dropout']),
          # --- Second Convolutional Block ---
          tw.keras.layers.Conv2D(self.hparams['n_filters'] * 2,
                                 kernel_size=(1, self.hparams['s_filters']),
                                 padding='same',
                                 kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.Conv2D(self.hparams['n_filters'] * 2,
                                 kernel_size=(1, self.hparams['s_filters']),
                                 padding='same',
                                 kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
          tw.keras.layers.Dropout(self.hparams['dropout']),
          # --- Third Convolutional Block ---
          tw.keras.layers.Conv2D(self.hparams['n_filters'] * 4,
                                 kernel_size=(1, self.hparams['s_filters']),
                                 padding='same',
                                 kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.Conv2D(self.hparams['n_filters'] * 4,
                                 kernel_size=(1, self.hparams['s_filters']),
                                 padding='same',
                                 kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='same'),
          tw.keras.layers.Dropout(self.hparams['dropout']),
          # --- Flatten for Classification Task ---
          tw.keras.layers.Flatten(),
          # --- First FCN Layer ---
          tw.keras.layers.Dense(self.hparams['n_units'],
                                kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.Dropout(self.hparams['dropout']),
          # --- Second FCN Layer ---
          tw.keras.layers.Dense(self.hparams['n_units'] * 2,
                                kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.Dropout(self.hparams['dropout']),
          # --- Third FCN Layer ---
          tw.keras.layers.Dense(self.hparams['n_units'] * 2,
                                kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          tw.keras.layers.Dropout(self.hparams['dropout']),
          # --- Fourth FCN Layer ---
          tw.keras.layers.Dense(self.hparams['n_units'] * 2,
                                kernel_initializer=self.hparams['m_init']),
          tw.keras.layers.BatchNormalization(),
          self.hparams['activ_fn'],
          # --- Final FCN Prediction Layer ---
          tw.keras.layers.Dense(20, 'softmax')
        ], name='TCNN')

        model_tcnn.compile(optimizer=self.hparams['optim'](),
                          metrics=['categorical_accuracy'],
                          loss='categorical_crossentropy')

        return model_tcnn

    def periodic_rate_decrementer(self, start: int, period: int, factor: float) -> Callable[[int, float], float]:
        """Decrease the loss by a factor every period epochs starting from a point."""
        def stabilize_lr(epoch: int, learn: float) -> float:
            if epoch < start:
                return learn
            if (epoch - start) % period == 0:
                return learn * factor
            return learn
        return stabilize_lr

    def epoch_graph_callback(self) -> tw.keras.callbacks.Callback:
        """Allow log output to be inspected using TensorBoard."""
        return tw.keras.callbacks.TensorBoard(log_dir=self.logs_dir_,
                                              write_graph=True)

    def gpu_utiliz_callback(self) -> tw.keras.callbacks.Callback:
        """Watch out for GPU memory / utilization constraints."""
        return GpuStatsLogger(['/gpu:0'], memory_utilization=True, gpu_utilization=True)

    def lr_updater_callback(self) -> tw.keras.callbacks.Callback:
        return tw.keras.callbacks.LearningRateScheduler(verbose=1,
          schedule=self.periodic_rate_decrementer(*self.hparams['lr_update']))

    def fitting_callbacks(self) -> List[tw.keras.callbacks.Callback]:
        return [
          self.epoch_graph_callback(),
          self.gpu_utiliz_callback(),
          self.lr_updater_callback()
        ]


class AttentionTCNN(Model[AttentionTCNNParams]):
    def __init__(self, hparams: AttentionTCNNParams, device: torch.device, verbose: bool = True):
        super().__init__(hparams)

        # Parameterize the training process
        self.n_workers = 0
        self.n_prefetch = 2
        self.device = device
        self.verbose = verbose

        # Parameterize the model
        self.in_chan = 1
        self.out_chan = 20
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        # Create the network and send it to the provided device to enhance speed
        self.model_ = TemporalAttentionNN(self.in_chan, self.out_chan,
                                          bottleneck=hparams['bottleneck'],
                                          s_filters=hparams['s_filters'],
                                          n_filters=hparams['n_filters'],
                                          init_fn=hparams['init_fn'],
                                          activ_fn=hparams['activ_fn'],
                                          n_units=hparams['n_units'],
                                          dropout=hparams['dropout'],
                                          norm=hparams['norm'],
                                          bias=hparams['bias'],
                                          device=device,
                                          verbose=verbose).to(self.device)

    def fit(self, train_data: ny.ndarray, train_labels: ny.ndarray,
                  valid_data: ny.ndarray = None, valid_labels: ny.ndarray = None) -> HistoryProgress:
        # Prepare data for processing
        train_loader, \
        valid_loader = self.__get_loaders(train_data, train_labels, \
                                          valid_data, valid_labels)

        # Instantiate the optimizer using the model's params
        opt = self.hparams['optim'](self.model_.parameters())

        # Optionally use a scheduler to improve the stability & convergence of the training
        if 'sch_lr' in self.hparams and self.hparams['sch_lr'] is not None:
            sch = self.hparams['sch_lr'](opt, self.verbose)
        else:
            sch = None

        # Train the network and evaluate it over each epoch
        history = self.model_.fit(train_loader, valid_loader,
                                  loss_fn=self.loss_fn,
                                  optim=opt,
                                  scheduler=sch,
                                  n_epochs=self.hparams['n_epochs'])

        # Expose the history of the training (and validation)
        return TorchHistoryProgress(history)

    def predict(self, data: ny.ndarray) -> ny.ndarray:
        loader = self.__get_loader(data, shuffle=False)
        y_hat = self.model_.predict(loader, with_labels=False)
        return y_hat

    def __get_loaders(self, train_data: ny.ndarray, train_labels: ny.ndarray,
                      valid_data: ny.ndarray = None, valid_labels: ny.ndarray = None) -> Tuple[data.DataLoader, data.DataLoader]:
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
        train_loader = self.__get_loader(train_data, train_labels)
        valid_loader = self.__get_loader(valid_data, valid_labels) \
                       if valid_data is not None and valid_labels is not None else None
        return train_loader, valid_loader

    def __get_loader(self, data: ny.ndarray, labels: ny.ndarray | None = None, shuffle: bool = True) -> data.DataLoader:
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
        loader = DataLoader(dataset, batch_size=self.hparams['n_batch'],
                            shuffle=shuffle, num_workers=self.n_workers,
                            pin_memory=True, prefetch_factor=self.n_prefetch)
        return loader
