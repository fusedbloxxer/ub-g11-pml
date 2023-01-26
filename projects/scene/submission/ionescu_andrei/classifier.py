from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics
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


class UnsupervisedClassifier(abc.ABC, BaseEstimator, ClassifierMixin):
  def __init__(self,
               model: cluster.AffinityPropagation | cluster.DBSCAN,
               n_classes: int = 6,
               f_extractor: FeatureExtractor = ColorHistFeatureExtractor(),
               silhouette_dist: str = 'euclidian',
               verbose: bool = True) -> None:
    super().__init__()

    # Internal params
    self.f_extractor = f_extractor
    self.silhouette_dist = silhouette_dist
    self.model = model
    self.verbose = verbose
    self.n_classes = n_classes
    self.n_clusters: int = None

  def evaluate(self, data: ny.ndarray, labels: ny.ndarray) -> ny.ndarray:
    """Expect an array of unprocessed images of size (N, C, H, W) and infer
    the best cluster assignments to match the given labels."""
    return self._clusters_to_labels(labels, self.predict(data))

  def predict(self, data: ny.ndarray) -> ny.ndarray:
    """Expect an array of unprocessed images of size (N, C, H, W) and infer
    the possible cluster assignments."""
    data_features = self.f_extractor.transform(data)

    # Predict the clusters using the obtained representation
    data_clusters = self._predict_clusters(data_features)

    # Return the cluster assignments for each data_sample
    return data_clusters

  def score(self, data: ny.ndarray, labels: ny.ndarray = None) -> float:
    """Expect an array of unprocessed images of size (N, C, H, W) and estimate
    a score on the unseen data."""
    data_features = self.f_extractor.transform(data)

    # Predict the clusters using the obtained representation
    data_clusters = self._predict_clusters(data_features)

    # Compute estimated scores
    scores = {}

    # Use silhouette to determine if the clustering is good
    if self.n_clusters > 1:
      silhouette: float = metrics.silhouette_score(data_features, data_clusters, metric=self.silhouette_dist)
      scores['silhouette'] = silhouette
    else:
      scores['silhouette'] = -1.0

    if labels is not None:
      pred_labels: ny.ndarray = self._clusters_to_labels(labels, data_clusters)
      accuracy: float = metrics.accuracy_score(labels, pred_labels)
      scores['accuracy'] = accuracy
    else:
      scores['accuracy'] = -1.0

    # Aggregate the scores
    return scores

  def fit(self, data: ny.ndarray, labels: ny.ndarray = None) -> 'UnsupervisedClassifier':
    """Build a representation on the given data, and extract features based on it.
    Expects an array of size (N, C, H, W) of unprocessed images."""
    if labels is not None and data.shape[0] != labels.shape[0]:
      raise ValueError('Data and labels shape mismatch')

    # Train the feature extractor to build a better representation over the raw data
    self.f_extractor.fit_transform(data)

    # Obtain the new representation
    data_features = self.f_extractor.transform(data)

    # Cluster over the computed features
    self.model.fit(data_features, labels)

    # Save the number of clusters found
    self.n_clusters = ny.max(self.model.labels_) + 1

    # Conform to API
    return self

  def _clusters_to_labels(self, labels: ny.ndarray, data_clusters: ny.ndarray) -> ny.ndarray:
    """Having an array of size (N,) of cluster assignments, reassign them
    to labels (N,) in order to maximize the metric."""
    # Get the unique indicators
    # If the number of clusters matches the number of labels, we have optimum solution
    if self.n_clusters == 0:
      return data_clusters
    elif ny.equal(self.n_classes, self.n_clusters):
      m_conf: ny.ndarray = metrics.confusion_matrix(labels, data_clusters)

      # Find the pairs for the optimum cost
      r_i, c_i = linear_sum_assignment(m_conf, maximize = True)

      # Reassign the clusters to the corresponding labels
      cluster_to_label = {x_from: x_to for x_from, x_to in zip(c_i, r_i)}
      assignments = ny.vectorize(lambda x: cluster_to_label[x])(data_clusters)
      return assignments
    else: # Otherwise we assign clusters to multiple labels
      m_conf = ny.zeros((self.n_classes, self.n_clusters))

      # Add all entries according by pairing them
      for y_truth, y_cluster in zip(labels, data_clusters):
        m_conf[y_truth][y_cluster] += 1.

      # Obtain cluster-label mappings
      cluster_to_label = {}
      for i_cluster in range(self.n_clusters):
        cluster_to_label[i_cluster] = m_conf[:, i_cluster].argmax(axis=0)

      # Reassign the clusters to the corresponding labels
      assignments = ny.vectorize(lambda x: cluster_to_label[x])(data_clusters)
      return assignments

  @abc.abstractmethod
  def _predict_clusters(self, data_features: ny.ndarray) -> ny.ndarray:
    """Given an array of preprocessed features (N, F) apply clustering and
    predict the associated clusters."""
    raise NotImplementedError()


class AffinityPropagationClassifier(UnsupervisedClassifier):
  def __init__(self,
               damping: int = 0.5,
               n_classes: int = 6,
               f_extractor: FeatureExtractor = ColorHistFeatureExtractor(),
               silhouette_dist: str = 'euclidean',
               verbose: bool = True) -> None:
    # Internal params
    self.damping = damping
    model = AffinityPropagation(damping=damping)

    # Go to parent and resolve
    super().__init__(model=model,
                     n_classes=n_classes,
                     f_extractor=f_extractor,
                     silhouette_dist=silhouette_dist,
                     verbose=verbose)

  def _predict_clusters(self, data: ny.ndarray) -> ny.ndarray:
    return self.model.predict(data)


class DBSCANClassifier(UnsupervisedClassifier):
  def __init__(self,
               n_classes: int = 6,
               f_extractor: FeatureExtractor = ColorHistFeatureExtractor(),
               silhouette_dist: str = 'euclidean',
               verbose: bool = True,
               eps: float = 0.5,
               min_samples: int = 5,
               metric: str = 'euclidean',
               p: int | str = 2) -> None:
    # Internal params
    self.eps = eps
    self.min_samples = min_samples
    self.metric = metric
    self.p = p
    model = DBSCAN(eps=eps,
                           min_samples=min_samples,
                           metric=metric,
                           p=p)

    # Go to parent and resolve
    super().__init__(model=model,
                     n_classes=n_classes,
                     f_extractor=f_extractor,
                     silhouette_dist=silhouette_dist,
                     verbose=verbose)

  def fit(self, data: ny.ndarray, labels: ny.ndarray = None) -> 'UnsupervisedClassifier':
    if labels is not None and data.shape[0] != labels.shape[0]:
      raise ValueError('Data and labels shape mismatch')

    # Train the feature extractor to build a better representation over the raw data
    self.f_extractor.fit_transform(data)

    # Obtain the new representation
    data_features = self.f_extractor.transform(data)

    # Cluster over the computed features
    self.model.fit(data_features, labels)

    # Save the number of clusters found
    self.n_clusters = ny.max(self.model.labels_) + 1

    # Save the corepoints for inference caching
    self._core_points: typing.Dict[int, ny.ndarray] = {}

    # Retrieve the corepoints
    for i_cluster in range(self.n_clusters):
      i_core: ny.ndarray = self.model.labels_ == i_cluster
      cluster_corepoints = data_features[i_core]
      self._core_points[i_cluster] = cluster_corepoints

    # Conform to API
    return self

  def _predict_clusters(self, data_features: ny.ndarray) -> ny.ndarray:
    pred_clusters: typing.List[float] = []
    for sample_features in data_features:
      cluster_min_dist: typing.List[ny.ndarray] = []

      # Compute the minimum distances from a sample to each existing cluster
      for i_cluster, cluster_corepoints in self._core_points.items():
        dist = ny.sum((sample_features - cluster_corepoints) ** 2, axis=1)
        cluster_min_dist.append(ny.min(dist).item())

      # Choose the closest corepoint's cluster as the prediction
      if len(cluster_min_dist) == 0:
        i_closest_cluster = -1
      else:
        i_closest_cluster: int = ny.argmin(cluster_min_dist)
      pred_clusters.append(i_closest_cluster)
    return ny.array(pred_clusters)


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
    self._model = TV.models.get_model('vgg11', weights=TV.models.VGG11_Weights.IMAGENET1K_V1)
    self._preprocess: torch.Module = TV.models.VGG11_Weights.IMAGENET1K_V1.transforms()
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

  def __str__(self) -> str:
    return f'FE_TL'

  def __repr__(self) -> str:
    return f'FE_TL'

