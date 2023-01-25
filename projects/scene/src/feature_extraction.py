from sklearn.cluster import MiniBatchKMeans, AffinityPropagation
from skimage.feature import hog
import numpy as ny
import typing
import abc
import cv2


from data import preprocess


class FeatureTransformer(abc.ABC):
  def __init__(self) -> None:
    super().__init__()

  @abc.abstractmethod
  def transform(self, image: ny.ndarray) -> ny.ndarray:
    """Receive an unprocessed image of size (C, H, W) and extract features,
    outputiing an array of size (D, E), where D is the number of descriptors,
    and each descriptor has an embedding of size E."""
    raise NotImplementedError()


class SIFTFeatureTransformer(FeatureTransformer):
  def __init__(self, visualize: bool = False,
               grid: typing.Tuple[int, int, int] = (4, 8, 8)) -> None:
    super().__init__()

    # Internal params
    self._grid = grid
    self._visualize = visualize
    self._transformer = cv2.SIFT_create()

  def transform(self, image: ny.ndarray) -> typing.Tuple[typing.Tuple[cv2.KeyPoint, ...], ny.ndarray, ny.ndarray, ny.ndarray] \
                                          | ny.ndarray:
    # Follow OpenCV standard HxWxC
    raw_image = image.transpose((1, 2, 0))
    H, W, C = raw_image.shape

    # Use grayscale in order to apply the transformation
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)

    # Create grid of keypoints in order to have the same output for each image
    keypoints: typing.List[cv2.KeyPoint] = []
    offset, stride, diameter = self._grid
    for i in range(offset, H, stride):
      for j in range(offset, W, stride):
        keypoints.append(cv2.KeyPoint(i, j, diameter))

    # Apply the transformer to obtain descriptors
    detection_result = self._transformer.compute(gray_image, keypoints)
    kpts: typing.Tuple[cv2.KeyPoint, ...] = detection_result[0]
    desc: ny.ndarray = detection_result[1]

    # Return the extracted features
    if self._visualize:
      return kpts, desc, gray_image, raw_image
    else:
      return desc


class HOGFeatureTransformer(FeatureTransformer):
  def __init__(self, visualize: bool = False, grayscale: bool = False) -> None:
    super().__init__()

    # Internal Params
    self._grayscale = grayscale
    self._visualize = visualize

  def transform(self, image: ny.ndarray) -> ny.ndarray:
    # Add temporary batch_axis
    raw_image = image[None, ...]

    # Optionally use grayscale before applying the transformer
    proc_image = preprocess(raw_image, grayscale=self._grayscale, min_max=False)
    proc_image = proc_image.squeeze((0,)).transpose((1, 2, 0))

    # Apply the transformer to obtain image descriptors
    detection_result = hog(proc_image, feature_vector=False, channel_axis=2, visualize=self._visualize)

    # Extract the feature descriptors
    if self._visualize:
      features: ny.ndarray = detection_result[0]
    else:
      features: ny.ndarray = detection_result

    # Flatten the features and get the descriptors for each block
    features = ny.reshape(features, (features.shape[0] ** 2, -1))

    if self._visualize:
      return (features, detection_result[1])
    else:
      return features


class FeatureExtractor(abc.ABC):
  def __init__(self,
               grayscale: bool = False,
               min_max: bool = True):
    super().__init__()

    # Configure common image representation parameters
    self._grayscale = grayscale
    self._min_max = min_max

  @abc.abstractmethod
  def fit(self, data: ny.ndarray, labels: ny.ndarray) -> None:
    """Build a representation on the given data, and extract features based on it.
    Expects an array of size (N, C, H, W) of unprocessed images and (N,) labels."""
    raise NotImplementedError()

  @abc.abstractmethod
  def features(self, data: ny.ndarray) -> ny.ndarray:
    """Receive an array of raw unprocessed images (N, C, H, W) and extract features,
    outputting an array of size (N, F)."""
    raise NotImplementedError()

  @abc.abstractmethod
  def image_features(self, image: ny.ndarray) -> ny.ndarray:
    """Receive an unprocessed image of size (C, H, W) and extract features,
    outputiing an array of size (F,)"""
    raise NotImplementedError()

  def preprocess(self, data: ny.ndarray) -> ny.ndarray:
    return preprocess(data, grayscale=self._grayscale, min_max=self._min_max)


class ColorHistFeatureExtractor(FeatureExtractor):
  def __init__(self,
               intervals: int = 256,
               density: bool = True,
               range: typing.Tuple[int, int] = None,
               **kwargs):
    super().__init__(**kwargs)

    # Param validation
    if range is None:
      range = (0, 1) if self._min_max else (0, 255)

    # Internal params
    self._intervals = intervals
    self._range = range
    self._density = density

  def fit(self, data: ny.ndarray, labels: ny.ndarray) -> None:
    """Dummy method. Does not need to build a representation, instances are enough."""
    return None

  def features(self, data: ny.ndarray) -> ny.ndarray:
    # Obtain the features based on the color histograms
    features = []

    # Iterative processsing over each image
    for image in data:
      features.append(self.image_features(image))

    # Bundle all features
    return ny.stack(features, axis=0)

  def image_features(self, image: ny.ndarray) -> ny.ndarray:
    # Preprocess the current image
    image = self.preprocess(image[None, ...]).squeeze(0)

    # Extract the expected shapes for the input image
    C, H, W = image.shape

    # Used to cache an image's histograms
    img_hist = []

    # Iterate over color channels and compute each histogram while retaining them
    for chan in range(C):
      # Obtain the color histogram for a certain channel
      hist_result: typing.Tuple[ny.ndarray, typing.Any] = ny.histogram(image[chan],
                                                                       bins=self._intervals,
                                                                       range=self._range,
                                                                       density=False)
      # Extract the corresponding values
      chan_hist: ny.ndarray = hist_result[0]

      if self._density:
        # Normalize the histogram
        chan_hist: ny.ndarray = chan_hist / hist_result[0].sum()

      # Save the channel_hist
      img_hist.append(chan_hist)

    # Concat an image's channel histograms
    img_hist = ny.concatenate(img_hist, axis=0)
    return img_hist


class BOVWFeatureExtractor(FeatureExtractor):
  def __init__(self,
               n_features: int,
               transformer: FeatureTransformer,
               batch_size: int = 1024,
               norm: bool = True,
               seed: int = 77):
    super().__init__(grayscale=True, min_max=True)

    # Internal params
    self._norm = norm
    self._n_features = n_features
    self._batch_size = batch_size
    self._transformer = transformer
    self._cluser_algo = MiniBatchKMeans(n_clusters=n_features,
                                                         batch_size=batch_size,
                                                         random_state=seed,
                                                         n_init='auto',
                                                         compute_labels=False)

  def fit(self, data: ny.ndarray, labels: ny.ndarray) -> None:
    # Transform all images to a new representation (N * D, E)
    descriptors: typing.List[ny.ndarray] = []

    # Iterate and transform each image
    for image in data:
      # (C, H, W) to (D, E)
      image_descriptors = self._transformer.transform(image)

      # Add all descriptors
      descriptors.append(image_descriptors)

    # Concatenate all descriptors for all images and find centroids
    descriptors: ny.ndarray = ny.concatenate(descriptors, axis=0)

    # Cluster the descriptors to find relevant visual words
    self._cluser_algo.fit(descriptors)

    # Retrieve the centroids and use them as the base vocabulary
    self.bovw_vocab_: ny.ndarray = self._cluser_algo.cluster_centers_

  def features(self, data: ny.ndarray) -> ny.ndarray:
    # Compute the image descriptors for each input: (N * D, E)
    descriptors: typing.List[ny.ndarray] = []
    same_size = None

    # Iterate and describe each image
    for image in data:
      # (C, H, W) to (D, E)
      image_descriptors = self._transformer.transform(image)

      # Assert that all images have descriptors of equals sizes!
      if same_size is None:
        same_size = image_descriptors.shape
      elif not ny.equal(same_size, image_descriptors.shape).all():
        raise Exception(f'Descriptors don\'t have the same size. Expected {same_size}, got {image_descriptors.shape}')

      # Keep all descriptors
      descriptors.append(image_descriptors)

    # Concatenate all descriptors for all images for faster processing
    descriptors: ny.ndarray = ny.concatenate(descriptors, axis=0)

    # Predict the cluster assignments in order to compute the bovw hists
    cluster_memberships = self._cluser_algo.predict(descriptors)
    cluster_memberships = ny.split(cluster_memberships, data.shape[0])
    cluster_memberships = ny.array(cluster_memberships)

    # Compute the bovw histograms
    hist_count: ny.ndarray = ny.apply_along_axis(ny.bincount, axis=1, arr=ny.array(cluster_memberships), minlength=256)

    # Optionally normalize the histogram
    if self._norm:
      return hist_count / hist_count.sum(axis=1)[:, None]
    else:
      return hist_count

  def image_features(self, image: ny.ndarray) -> ny.ndarray:
    # Describe the image using the learnt clusters from fit function
    image_descriptors: ny.ndarray = self._transformer.transform(image)
    cluster_centers: ny.ndarray = self._cluser_algo.predict(image_descriptors)
    hist_count: ny.ndarray = ny.bincount(cluster_centers, minlength=self._n_features)

    # Optionally normalize the histogram
    if self._norm:
      return hist_count / hist_count.sum()
    else:
      return hist_count

