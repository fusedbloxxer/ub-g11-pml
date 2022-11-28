from __future__ import annotations
from typing import TypedDict, TypeVar, Callable, Generic, List, Any, Tuple
import abc
import pathlib as pb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from nvitop.callbacks.keras import GpuStatsLogger
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as pt
import numpy as ny
import sklearn as sn
import tensorflow as tw


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
    dropout: float
    n_units: int
    n_batch: int
    n_epochs: int


class SVMParams(HyperParams):
    C: float
    kernel: str
    gamma: str


class KNNParams(HyperParams):
    n_neighbors: int
    p: int


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
            train_labels: (n_samples, 1)
            valid_data: (n_samples, n_features)
            valid_labels: (n_samples, 1)
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
