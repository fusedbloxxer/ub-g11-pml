from __future__ import annotations
from typing import TypeVar, Callable, Generic, List, Any, Tuple
import abc
import pathlib as pb
from nvitop.callbacks.keras import GpuStatsLogger
import torch.nn as nn
import torch
import tensorflow as tw
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import sklearn as sn
import numpy as ny
from processing import get_loader, get_loaders
from progress import HistoryProgress, NNHistoryProgress, TorchHistoryProgress, TrainValidHistoryProgress
from params import AutoEncoderParams, KNNParams, SVMParams, BoostedTreesParams
from params import TCNNParams, AttentionTCNNParams
from blocks import AutoEncoder, TemporalAttentionNN


HP = TypeVar('HP', bound=Any)
class Model(abc.ABC, Generic[HP]):
    """Base Model class used to abstract various implementations in different
    frameworks. Each deriving model has to implement both a fitting and a prediction
    function. Note: HP represents the HyperParameters of the model."""
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
        """Predict an array of some sort (ex. embeddings/labels) for the given data."""
        raise NotImplementedError()


class ClassifierModel(Model[HP], Generic[HP]):
    """Abstract model class which adds two helper methods to the base model class:
        - accuracy - computes an accuracy after predicting the labels
        - conf - computes a confusion matrix after predicting the labels"""
    def __init__(self, hparams: HP):
        super().__init__(hparams)

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
class SKModel(ClassifierModel[HP], Generic[HP, TSKModel]):
    """Base model class used to abstract away implementation details of
    scikit-learn models. Because the SKLearn framework uses common methods
    for various algorithms, the fitting and prediction functions are alike
    for many of them."""
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
    """Boosted Trees SKLearn Classifier"""
    def __init__(self, hparams: BoostedTreesParams, verbose: int = 0):
        super().__init__(hparams, GradientBoostingClassifier(
            learning_rate=hparams['learning_rate'],
            n_estimators=hparams['n_estimators'],
            subsample=hparams['subsample'],
            max_depth=hparams['max_depth'],
            verbose=verbose,
        ))


class SVMModel(SKModel[SVMParams, SVC]):
    """SVM SKLearn Classifier"""
    def __init__(self, hparams: SVMParams, verbose: bool = True):
        super().__init__(hparams=hparams, model=SVC(C=hparams['C'],
                                                    kernel=hparams['kernel'],
                                                    gamma=hparams['gamma'],
                                                    verbose=verbose))


class KNNModel(SKModel[KNNParams, KNeighborsClassifier]):
    """KNN SKLearn Classifier"""
    def __init__(self, hparams: KNNParams):
        super().__init__(hparams=hparams, model=KNeighborsClassifier(
            n_neighbors=hparams['n_neighbors'],
            p=hparams['p'],
        ))


class TCNNModel(ClassifierModel[TCNNParams]):
    """Temporal Convolutional Neural Network Model Classifier written using Tensorflow.

    Receives data of size (N, C, 1, S) and applies convolutions over the temporal
    axis S, then trains some fully-connected layers on top to predict the user."""
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
          # --- Add small amount of noise ---
          tw.keras.layers.GaussianNoise(self.hparams['noise_std']),
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
        # Update the learning rate on each epoch using custom scheduling
        return tw.keras.callbacks.LearningRateScheduler(verbose=1,
          schedule=self.periodic_rate_decrementer(*self.hparams['lr_update']))

    def fitting_callbacks(self) -> List[tw.keras.callbacks.Callback]:
        # Use multiple callbacks after each epoch
        return [
          self.epoch_graph_callback(),
          self.gpu_utiliz_callback(),
          self.lr_updater_callback()
        ]


class AttentionTCNNModel(ClassifierModel[AttentionTCNNParams]):
    """Attention-based Temporal Convolutional Neural Network Model Classifier
    written using PyTorch.

    Receives data of size (N, C, 1, S) and applies residual-convolutions over the temporal
    axis S followed by SelfAttentionModules, then trains some fully-connected layers on
    top to predict the user."""
    def __init__(self, hparams: AttentionTCNNParams, device: torch.device, verbose: bool = True):
        super().__init__(hparams)

        # Parameterize the training process
        self.n_workers = 0
        self.n_prefetch = 2
        self.device = device
        self.verbose = verbose

        # Parameterize the model
        self.in_chan = 3
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
        valid_loader = get_loaders(train_data, train_labels, \
                                   valid_data, valid_labels, \
                                   shuffle=True, batch=self.hparams['n_batch'],
                                   workers=self.n_workers, prefetch=self.n_prefetch)

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
        loader = get_loader(data, shuffle=False, batch=self.hparams['n_batch'], \
                            workers=self.n_workers, prefetch=self.n_prefetch)
        y_hat = self.model_.predict(loader, with_labels=False)
        return y_hat


class AutoEncoderModel(Model[AutoEncoderParams]):
    """Convolutional AutoEncoder which produces sparse embeddings in lower dimensions
    in order to better represent the input data and remove noise."""
    def __init__(self, hparams: AutoEncoderParams, device: torch.device | None = None,
                 verbose: bool = False) -> None:
        super().__init__(hparams)

        # Training params
        self.device = device
        self.verbose = verbose
        self.prefetch = 2
        self.workers = 0

        # Parameterize the model
        self.in_chan = 3

        # Instantiate the model to use it later for training & prediction
        self.model_ = AutoEncoder(in_chan=self.in_chan,
                                  filter_chan=self.hparams['n_filters'],
                                  embedding_features=self.hparams['embedding_features'],
                                  activ_fn=self.hparams['activ_fn'](),
                                  dropout=self.hparams['dropout'],
                                  init_fn=self.hparams['init_fn'],
                                  bias=self.hparams['bias'],
                                  device=device,
                                  verbose=verbose).to(self.device)

    def fit(self, train_data: ny.ndarray, train_labels: ny.ndarray,
                  valid_data: ny.ndarray = None, valid_labels: ny.ndarray = None) -> HistoryProgress:
        # Prepare data for processing
        train_loader, \
        valid_loader = get_loaders(train_data, train_labels, \
                                   valid_data, valid_labels, \
                                   shuffle=True, batch=self.hparams['n_batch'], \
                                   workers=self.workers, prefetch=self.prefetch)

        # Instantiate the optimizer using the model's params
        opt = self.hparams['optim'](self.model_.parameters())

        # Optionally use a scheduler to improve the stability & convergence of the training
        if 'sch_lr' in self.hparams and self.hparams['sch_lr'] is not None:
            sch = self.hparams['sch_lr'](opt, self.verbose)
        else:
            sch = None

        # Train the network and evaluate it over each epoch
        history = self.model_.fit(train_loader, valid_loader,
                                  loss_fn=self.hparams['loss_fn'],
                                  optim=opt, scheduler=sch,
                                  n_epochs=self.hparams['n_epochs'])

        # Expose the history of the training (and validation)
        return TorchHistoryProgress(history)

    def predict(self, data: ny.ndarray, with_loss: bool = False) -> ny.ndarray | Tuple[ny.ndarray, float]:
        """Map the data to a lower representation called embeddings."""
        loader = get_loader(data, shuffle=False, batch=self.hparams['n_batch'], \
                            workers=self.workers, prefetch=self.prefetch)
        if with_loss:
            embeddings, loss = self.model_.predict(loader, with_labels=False, \
                                                   loss_fn=self.hparams['loss_fn'])
            return embeddings.numpy(), loss
        return self.model_.predict(loader, with_labels=False).numpy()


class HybridAutoEncoderClassifier(ClassifierModel[None], Generic[HP]):
    """Hybrid classifier model which trains a generic classifier on top of
    data embeddings produced by a CNN AutoEncoder."""
    def __init__(self, hpa_model: AutoEncoderModel,
                       hpc_model: ClassifierModel[HP],
                       verbose: bool = True) -> None:
        super().__init__(hparams=None)
        self.verbose = verbose
        self.ae_model = hpa_model
        self.cl_model = hpc_model

    def fit(self, train_data: ny.ndarray, train_labels: ny.ndarray,
                  valid_data: ny.ndarray = None, valid_labels: ny.ndarray = None) -> HistoryProgress:
        # First stage of training - Learn embeddings
        hpa_hist = self.ae_model.fit(train_data, train_labels, \
                                     valid_data, valid_labels)
        if self.verbose:
            hpa_hist.show()

        # Transform the data into embeddings
        train_embeddings = self.ae_model.predict(train_data)
        valid_embeddings = None if valid_data is None else self.ae_model.predict(valid_data)

        # Second stage of training - Train classifier on top of embeddings
        hpc_hist = self.cl_model.fit(train_embeddings, train_labels, \
                                     valid_embeddings, valid_labels)

        # Return the history of the classifier
        return hpc_hist

    def predict(self, data: ny.ndarray) -> ny.ndarray:
        """Predict an array of labels for the given data."""
        if self.verbose:
            embeddings, loss = self.ae_model.predict(data, with_loss=True)
            print(f'Mean Reconstruciton Loss: {loss:.3f}')
        else:
            embeddings = self.ae_model.predict(data)

        # Predict using the generated embeddings
        return self.cl_model.predict(embeddings)

