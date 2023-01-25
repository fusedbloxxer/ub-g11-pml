from sklearn.cluster import DBSCAN, AffinityPropagation
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision as TV
import torch.utils.data
import torch.nn as nn
import torch
import numpy as ny
import typing
import abc


from feature_extraction import FeatureExtractor
from data import SceneDataset


class Classifier(abc.ABC):
  def __init__(self) -> None:
    super().__init__()

  @abc.abstractmethod
  def fit(self, data: ny.ndarray, labels: ny.ndarray) -> None:
    """Build a representation on the given data, and extract features based on it.
    Expects an array of size (N, C, H, W) of unprocessed images and (N,) labels."""
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(self, data: ny.ndarray) -> ny.ndarray:
    """Expect an array of unprocessed image of size (N, C, H, W) and predict
    labels, outputting (N,) elements."""
    raise NotImplementedError()


class UnsupervisedClassifier(Classifier):
  def __init__(self) -> None:
    super().__init__()

  @abc.abstractmethod
  def predict_clusters(self, data: ny.ndarray) -> ny.ndarray:
    """Expect an array of unprocessed image of size (N, C, H, W) and predict
    the cluster indices, outputting (N,) elements."""
    raise NotImplementedError()


class TransferLearningClassifier(FeatureExtractor, Classifier):
  def __init__(self,
               n_classes: int,
               device: torch.device,
               epochs: int = 20,
               batch_size: int = 32,
               verbose: bool = True) -> None:
    FeatureExtractor.__init__(self, grayscale=False, min_max=False)
    Classifier.__init__(self)

    # Internal params
    self._n_classes = n_classes
    self._verbose = verbose
    self._device = device
    self._n_epochs = epochs
    self._batch_size = batch_size
    self._weights = TV.models.VGG11_Weights.IMAGENET1K_V1
    self._model = TV.models.get_model('vgg11', weights=self._weights)
    self._preprocess: torch.Module = self._weights.transforms()
    self._fn_loss = nn.CrossEntropyLoss()

    # Set initial state
    self._model.requires_grad_(False)
    self._model = self._model.eval()
    self._model = self._model.to(self._device)

  def fit(self, data: ny.ndarray, labels: ny.ndarray) -> None:
    # Use mini-batch loading in order to fit the data in-memory
    data_with_labels: Dataset = SceneDataset(data, labels)
    data_fetcher: DataLoader = DataLoader(data_with_labels,
                                          batch_size=self._batch_size,
                                          shuffle=True)

    # Configure Transfer Learning
    model = self._model.train(True)
    model.requires_grad_(False)
    model.classifier[ 3] = nn.Linear(4096, 4096, True, device=self._device)
    model.classifier[ 3].requires_grad_(True)
    model.classifier[-1] = nn.Linear(4096, self._n_classes, True, device=self._device)
    model.classifier[-1].requires_grad_(True)
    opt = torch.optim.Adam([
      *model.classifier[ 3].parameters(),
      *model.classifier[-1].parameters()
    ], lr=2e-4)

    # Finetune over the small given dataset
    for epoch_i in range(self._n_epochs):
      eacc = []
      for batch_i, (X_data, y_labels) in enumerate(data_fetcher):
        # User faster computing by selecting the given runtime device
        X_data, y_labels = X_data.to(self._device), y_labels.to(self._device)

        # Perform processing according to the specification given by the docs
        X_data: torch.Tensor = self.preprocess(X_data)

        # Train based on the given predictions
        opt.zero_grad()
        lgts: torch.Tensor = model.forward(X_data)
        class_loss: torch.Tensor = self._fn_loss(lgts, y_labels)
        class_loss.backward()
        opt.step()

        # Save running stats
        matches: torch.Tensor = torch.argmax(lgts, dim=1) == y_labels
        eacc.append(matches.clone().detach())

      # Display stats
      eacc: torch.Tensor = torch.cat(eacc)
      eacc = eacc.sum() / eacc.shape[0]
      print(f'epoch: {epoch_i} - accuracy: {eacc:.2f}')

    # Switch back to prediction mode in order to retrieve relevant features
    model = model.eval()
    model.requires_grad_(False)

  def fit_transform(self, data: ny.ndarray, labels: ny.ndarray) -> None:
    return self.fit(data, labels)

  def predict(self, data: ny.ndarray) -> ny.ndarray:
    # Use mini-batch loading for the input data that has no labels
    data_no_labels = TensorDataset(torch.tensor(data))
    data_fetcher: DataLoader = DataLoader(data_no_labels,
                                          batch_size=self._batch_size,
                                          shuffle=False)

    # Save predicted labels
    m_pred: typing.List[int] = []
    self._model.requires_grad_(False)
    self._model = self._model.eval()

    # Iterative prediction process
    for batch_i, (X_data,) in enumerate(data_fetcher):
      # Preprocess according to specification
      X_data: torch.Tensor = self.preprocess(X_data)

      # Forward pass while saving activations by hooks usage
      lgts: torch.Tensor = self._model.forward(X_data)
      s_pred: torch.Tensor = torch.argmax(lgts, axis=1)
      m_pred.extend(s_pred.detach().cpu().tolist())

    # Concatenate and return results
    return ny.array(m_pred)

  def transform(self, data: ny.ndarray) -> ny.ndarray:
    # Use mini-batch loading for the input data that has no labels
    data_no_labels = TensorDataset(torch.tensor(data))
    data_fetcher: DataLoader = DataLoader(data_no_labels,
                                          batch_size=self._batch_size,
                                          shuffle=False)

    # Listen to activations
    self._model.requires_grad_(False)
    self._model = self._model.eval()
    self._model.classifier[ 3].register_forward_hook(self.intermediary_values())

    # Extract intermediary activation one by one and save them
    int_activs: typing.List[ny.ndarray] = []

    # Iterative prediction process
    for batch_i, (X_data,) in enumerate(data_fetcher):
      # Preprocess according to specification
      X_data: torch.Tensor = self.preprocess(X_data)

      # Forward pass while saving activations by hooks usage
      self._model.forward(X_data)
      int_activs.append(self.int_activ_)

    # Concatenate and return results
    return ny.concatenate(int_activs, axis=0)

  def transform_image(self, image: ny.ndarray) -> ny.ndarray:
    # Prepare the input data
    image: torch.Tensor = self.preprocess(image)

    # Listen to activations
    self._model.requires_grad_(False)
    self._model = self._model.eval()
    self._model.classifier[ 3].register_forward_hook(self.intermediary_values())
    self._model.forward(image)
    activ: torch.Tensor = self.int_activ_
    return activ.squeeze(0).numpy()

  def preprocess(self, data: ny.ndarray | torch.Tensor) -> torch.Tensor:
    if not isinstance(data, torch.Tensor):
      data = torch.tensor(data)
    if data.ndim == 3:
      return self._preprocess(data.to(self._device).unsqueeze(0))
    else:
      return self._preprocess(data.to(self._device))

  def intermediary_values(self):
    def activ_callback(module: nn.Module, inp_data: torch.Tensor, out_result: torch.Tensor):
      self.int_activ_ = out_result.clone().detach().cpu().numpy()
    return activ_callback

