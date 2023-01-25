from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans
import sklearn.metrics as metrics
import sklearn.cluster as cluster
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision as TV
import torch.utils.data
import torch.nn as nn
import torch
import numpy as ny
from scipy.optimize import linear_sum_assignment
import scipy as sy
import typing
import abc


from feature_extraction import FeatureExtractor, ColorHistFeatureExtractor
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
    """Expect an array of unprocessed images of size (N, C, H, W) and predict
    labels, outputting (N,) elements."""
    raise NotImplementedError()


class UnsupervisedClassifier(abc.ABC):
  def __init__(self,
               model: cluster.AffinityPropagation | cluster.DBSCAN,
               f_extractor: FeatureExtractor = ColorHistFeatureExtractor(),
               silhouette_dist: str = 'euclidian',
               verbose: bool = True) -> None:
    super().__init__()

    # Internal params
    self._f_extractor = f_extractor
    self._silhouette_dist = silhouette_dist
    self._model = model
    self.verbose = verbose

  def evaluate(self, data: ny.ndarray, labels: ny.ndarray) -> ny.ndarray:
    """Expect an array of unprocessed images of size (N, C, H, W) and infer
    the best cluster assignments to match the given labels."""
    return self._clusters_to_labels(labels, self.predict(data))

  def predict(self, data: ny.ndarray) -> ny.ndarray:
    """Expect an array of unprocessed images of size (N, C, H, W) and infer
    the possible cluster assignments."""
    data_features = self._f_extractor.transform(data)

    # Predict the clusters using the obtained representation
    data_clusters = self._predict_clusters(data_features)

    # Return the cluster assignments for each data_sample
    return data_clusters

  def score(self, data: ny.ndarray) -> float:
    """Expect an array of unprocessed images of size (N, C, H, W) and estimate
    a score on the unseen data."""
    data_features = self._f_extractor.transform(data)

    # Predict the clusters using the obtained representation
    data_clusters = self._predict_clusters(data_features)

    # Compute an estimated score
    return metrics.silhouette_score(data_features, data_clusters, metric=self._silhouette_dist)

  def fit(self, data: ny.ndarray) -> None:
    """Build a representation on the given data, and extract features based on it.
    Expects an array of size (N, C, H, W) of unprocessed images."""
    # Train the feature extractor to build a better representation over the raw data
    self._f_extractor.fit_transform(data)

    # Obtain the new representation
    data_features = self._f_extractor.transform(data)

    # Cluster over the computed features
    self._model.fit(data_features)

  @abc.abstractmethod
  def _predict_clusters(self, data_features: ny.ndarray) -> ny.ndarray:
    """Given an array of preprocessed features (N, F) apply clustering and
    predict the associated clusters."""
    raise NotImplementedError()

  def _clusters_to_labels(self, labels: ny.ndarray, data_clusters: ny.ndarray) -> ny.ndarray:
    """Having an array of size (N,) of cluster assignments, reassign them
    to labels (N,) in order to maximize the metric."""
    m_conf: ny.ndarray = metrics.confusion_matrix(labels, data_clusters)

    # Find the pairs for the optimum cost
    r_i, c_i = linear_sum_assignment(m_conf, maximize = True)

    # Reassign the clusters to the corresponding labels
    cluster_to_label = {x_from: x_to for x_from, x_to in zip(c_i, r_i)}
    assignments = ny.vectorize(lambda x: cluster_to_label[x])(data_clusters)
    return assignments


class AffinityPropagationClassifier(UnsupervisedClassifier):
  def __init__(self,
               f_extractor: FeatureExtractor = ColorHistFeatureExtractor(),
               silhouette_dist: str = 'euclidian',
               verbose: bool = True,
               **kwargs) -> None:
    # Internal params
    model = KMeans(**kwargs)

    # Go to parent and resolve
    super().__init__(model, f_extractor, silhouette_dist, verbose)

  def _predict_clusters(self, data: ny.ndarray) -> ny.ndarray:
    return self._model.predict(data)


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

  def fit_transform(self, _: ny.ndarray) -> None:
    """Dummy method. Does not need to build another representation, instances are enough."""
    return None

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

