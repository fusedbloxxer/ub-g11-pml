from typing import Callable, Iterator, TypedDict, Type, Tuple, Generic, TypeVar
import torch.optim.optimizer as optimizer
import tensorflow as tw
import torch.nn as nn
import torch


class HyperParams(TypedDict):
  """Base class indicating the needed hyperparams to fine-tune a model during
  the training process. It's derived and used as a typed dict."""
  pass


class TrainParams(TypedDict):
  """Base class which indicates the training parameters to adjust how
  the training process should work."""
  n_folds: int


class SVMParams(HyperParams):
  """SKLearn SVC Model HyperParams"""
  C: float
  kernel: str
  gamma: str


class KNNParams(HyperParams):
  """SKLearn KNN Model HyperParams"""
  n_neighbors: int
  p: int


class BoostedTreesParams(HyperParams):
  """SKLearn BoostedTrees Model HyperParams"""
  learning_rate: float
  n_estimators: int
  subsample: float
  max_depth: int


class TCNNParams(HyperParams):
  """Tensorflow Temporal Convolutional Neural-Network HyperParams"""
  optim: Callable[[], tw.keras.optimizers.Optimizer]
  m_init: tw.keras.initializers.VarianceScaling
  activ_fn: tw.keras.layers.Activation
  lr_update: Tuple[int, int, float]
  noise_std: float
  n_filters: int
  s_filters: int
  dropout: float
  n_units: int
  n_batch: int
  n_epochs: int


class AttentionTCNNParams(HyperParams):
  """PyTorch Attention Temporal Convolutional Neural-Network HyperParams"""
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


class AutoEncoderParams(HyperParams):
  """PyTorch-based AutoEncoder HyperParams"""
  sch_lr: Callable[[optimizer.Optimizer], torch.optim.lr_scheduler._LRScheduler]
  optim: Callable[[Iterator[torch.nn.Parameter]], optimizer.Optimizer]
  init_fn: Callable[[torch.Tensor], None]
  activ_fn: Type[nn.Module]
  loss_fn: Type[nn.Module]
  embedding_features: int
  n_filters: int
  dropout: float
  n_epochs: int
  n_batch: int
  bias: bool


HPA = TypeVar('HPA', bound=AutoEncoderParams)
HPC = TypeVar('HPC', bound=HyperParams)
class AECHybridParams(Generic[HPA, HPC]):
  """AutoEncodingClassifierHybridParams - Contains params for both an
  AutoEncoder and for a Classifier"""
  hpa_params: HPA
  hpc_params: HPC

