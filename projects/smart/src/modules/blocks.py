import torch
import typing
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as FL
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler


class Swish(nn.Module):
  """Overcome the saturating gradients issue for the ReLU activation function
  by leaving a small window for gradients to flow on the negative part of the axis."""
  def __init__(self):
    super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RepeatModule(nn.Module):
  """Generate and chain modules n times in order to create deeper architectures."""
  def __init__(self, module_gen: typing.Callable[[], nn.Module], times: int):
    super(RepeatModule, self).__init__()

    # Inner params
    self.layers_ = nn.Sequential()

    # Generate N modules and chain them sequentially
    for i in range(times):
      self.layers_.add_module(name=f'module_{i}', module=module_gen())

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers_(x)


class ResModule2d(nn.Module):
  """Wrap an inner module between two bottleneck conversion blocks in order to
  speed up execution time while having an approximate loss, and offer a residual/identity
  path for gradient flow."""
  def __init__(self, in_chan: int, out_chan: int, module: nn.Module = None, bottleneck: int = 1, bias: bool = True):
    super(ResModule2d, self).__init__()

    # Internal params
    self.layers_ = nn.Sequential(module)
    self.bottleneck_ = bottleneck
    self.in_chan = in_chan
    self.out_chan = out_chan

    # Conditionally wrap the inner module between two bottleneck conversion blocks
    if bottleneck != 1:
      self.layers_.insert(0, nn.Conv2d(in_chan, in_chan // bottleneck, 1, bias=bias))
    if bottleneck != 1 or in_chan != out_chan:
      self.layers_.append(nn.Conv2d(in_chan // bottleneck, out_chan, 1, bias=bias))
    if in_chan != out_chan:
      self.res_ = nn.Conv2d(in_chan, out_chan, 1, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the inner module over the input data and add either the input
    to the output or a channel-modified version by using 1x1 conv."""
    if self.in_chan == self.out_chan:
      return x + self.layers_(x)
    else:
      return self.res_(x) + self.layers_(x)


class ResBlock2d(nn.Module):
  def __init__(self, in_chan: int, out_chan: int, kernel_size: int,
               activ_fn: nn.Module = nn.ReLU, norm: bool = True,
               bottleneck: int = 1, times: int = 3, bias: bool = True,
               dropout: float = 0.0):
    super(ResBlock2d, self).__init__()

    # Init hyperparams
    self.in_chan = in_chan
    self.out_chan = out_chan
    self.kernel_size = kernel_size
    self.bottleneck = bottleneck
    self.activ_fn = activ_fn
    self.dropout = dropout
    self.norm = norm
    self.bias = bias

    # Create layers
    module = RepeatModule(lambda: self.__create_block(), times)
    self.layers_ = ResModule2d(in_chan, out_chan, module, bottleneck)

  def __create_block(self) -> nn.Module:
    """Creates inner blocks from a set of conditionally applied operations. The
    convolution operation is applied across the temporal dimension."""
    layers = nn.Sequential()

    layers.append(nn.Conv2d(self.in_chan // self.bottleneck,
                            self.in_chan // self.bottleneck,
                            (1, self.kernel_size),
                            padding='same',
                            bias=self.bias))

    if self.norm is True:
      layers.append(nn.BatchNorm2d(self.in_chan // self.bottleneck))

    if self.activ_fn is not None:
      layers.append(self.activ_fn())

    if self.dropout is not None:
      layers.append(nn.Dropout2d(p=self.dropout))

    return layers

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers_(x)


class SelfAttentionBlock2d(nn.Module):
  def __init__(self, in_chan: int, bottleneck: int = 1, bias: bool = True) -> None:
    super().__init__()
    self.embed_chan = in_chan // bottleneck

    # Create single-attention layers
    self.conv_query = nn.Conv2d(in_chan, self.embed_chan, 1, bias=bias)
    self.conv_key = nn.Conv2d(in_chan, self.embed_chan, 1, bias=bias)
    self.conv_value = nn.Conv2d(in_chan, self.embed_chan, 1, bias=bias)
    self.conv_out = nn.Conv2d(self.embed_chan, in_chan, 1, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Receive an input of the following form size: (N, C, 1, S), where
    N is the mini-batch size, C is the number of channels and S is the length of
    the sequence and apply attention.

    Args:
        x (torch.Tensor): An input of size (N, C, 1, S).

    Returns:
        torch.Tensor: A new tensor on which selfAttention was applied according
        to the "Non-local Neural Networks" paper (2017, He et. al).
    """
    # Extract dimensions to enable reshaping later
    N, _, D, S = x.shape

    # Embed input using bottleneck
    query: torch.Tensor = self.conv_query(x)
    value: torch.Tensor = self.conv_value(x)
    key: torch.Tensor = self.conv_key(x)

    # Flatten temporal and spatial dimensions
    query = query.flatten(start_dim=2).transpose(1, 2)
    key = key.flatten(start_dim=2)
    value = value.flatten(start_dim=2).transpose(1, 2)

    # Compute attention map
    query_key = query @ key
    attention = torch.softmax(query_key, dim=2)

    # Apply attention over the input
    output = (attention @ value).transpose(1, 2).reshape(N, self.embed_chan, D, S)

    # Add residual connection
    return x + self.conv_out(output)


class GlobalMaxPool2d(nn.Module):
  """Apply MaxPool to reduce the spatial/temporal dimensions and provide
  the number of channels as features to linear units."""
  def __init__(self) -> None:
    super().__init__()

    # Internal layers
    self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    self.flatten = nn.Flatten(start_dim=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.flatten(self.pool(x))


class NNModule(nn.Module):
  """Base class for NeuralNetwork models. Offers support for weight init,
  and currently used device to hold the params."""
  def __init__(self, init_fn: typing.Callable[[torch.Tensor], None] = None,
               device: torch.device | None = None, verbose: bool = False,
               bias: bool = True):
    super().__init__()
    self._device = device
    self.verbose = verbose
    self.init_fn = init_fn
    self.bias = bias

  def _weights_initialization(self, module: nn.Module) -> None:
    # Define range of modules that will be weight-initialized according to init_fn
    allowed_types = tuple([
      nn.Linear,
      nn.Conv2d,
    ])

    # Apply the init_fn
    if isinstance(module, allowed_types):
      self.init_fn(module.weight)

      if self.bias:
        torch.nn.init.constant_(module.bias, 0.0)

  @property
  def device(self) -> torch.device:
    """Get the currently used device for the model or CPU by default."""
    if self._device is None:
      return torch.device('cpu')
    else:
      return self._device


class TemporalAttentionNN(NNModule):
  """Exploit the temporal/sequential nature of the input by using convolutions
  along this dimension and treat the spatial coordinates of the samples as
  channels in order to get better latent representations for each temporal point.
  Apply self-attention to keep global information along every step and use
  residual connections to improve the gradient flow during model optimization."""
  def __init__(self, in_chan: int, out_chan: int,
               s_filters: int = 3, n_filters: int = 128, n_units: int = 1024,
               activ_fn: nn.Module = Swish, norm: bool = True,
               bottleneck: int = 2, dropout: float = 0.3, bias: bool = True,
               init_fn: typing.Callable[[torch.Tensor], None] = None,
               device: torch.device = None, verbose: bool = False):
    super().__init__(init_fn=init_fn, device=device, verbose=verbose, bias=bias)

    # Internal params
    self.n_fcn_hidden_layers = 2
    self.n_cnn_hidden_layers = 2
    self.inner_repeat = 2
    self.bias = bias

    # --- Input Layer ---
    self.layers_ = nn.Sequential()

    # --- Stage 1 Residual Blocks ---
    self.layers_.add_module(name='stage_1', module=nn.Sequential(
      ResBlock2d(in_chan, n_filters, s_filters, activ_fn, norm,
                 1, bias, self.inner_repeat, dropout),
      SelfAttentionBlock2d(n_filters, bottleneck, bias),
      ResBlock2d(n_filters, n_filters, s_filters, activ_fn, norm,
                 bottleneck, bias, self.inner_repeat, dropout),
      nn.MaxPool2d((1, s_filters), (1, 2)),
    ))

    # --- Hidden Residual Blocks ---
    for i in range(self.n_cnn_hidden_layers):
      self.layers_.add_module(name=f'stage_{i + 2}', module=nn.Sequential(
        ResBlock2d(n_filters * 2 ** i, n_filters * 2 ** i, s_filters,
                   activ_fn, norm, bottleneck, bias, self.inner_repeat, dropout),
        SelfAttentionBlock2d(n_filters * 2 ** i, bottleneck, bias),
        ResBlock2d(n_filters * 2 ** i, n_filters * 2 ** (i + 1), s_filters,
                   activ_fn, norm, bottleneck, bias, self.inner_repeat, dropout),
        nn.MaxPool2d((1, s_filters), (1, 2)),
      ))

    # --- Reshaping the features ---
    self.layers_.append(GlobalMaxPool2d())

    # --- Input FCN Layer ---
    self.layers_.add_module(name='fcn_1', module=nn.Sequential(
      nn.Linear(n_filters * 2 ** self.n_cnn_hidden_layers, n_units, bias),
      nn.BatchNorm1d(n_units) if norm else nn.Identity(),
      activ_fn(),
      nn.Dropout(dropout)
    ))

    # --- Hidden FCN Layers ---
    for i in range(self.n_fcn_hidden_layers):
      self.layers_.add_module(name=f'fcn_{i + 2}', module=nn.Sequential(
        nn.Linear(n_units * 2 ** i, n_units * 2 ** (i + 1), bias),
        nn.BatchNorm1d(n_units * 2 ** (i + 1)) if norm else nn.Identity(),
        activ_fn(),
        nn.Dropout(dropout),
      ))

    # --- Output FCN Layer ---
    self.layers_.append(nn.Linear(n_units * 2 ** (self.n_fcn_hidden_layers), out_chan, bias))

    # Optionally initialize the weights with a custom approach
    if self.init_fn:
      self.apply(self._weights_initialization)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers_(x)

  def predict(self, loader: data.DataLoader, with_labels: bool = True,
              loss_fn: nn.Module | None = None) -> typing.Tuple[torch.Tensor, ...]:
    """Predict the list of labels for the given data (not a mini-batch). Do not SHUFFLE
    the dataset to be able to get the expected labels in the same initial order.

    Args:
        loader (data.DataLoader): A DataLoader wrapping up the current dataset.
        with_labels (bool): Specifies if labels are served in the loader (ex. for validation).
        loss_fn (torch.nn.Module): A loss functions that does not reduce the output.

    Returns:
        torch.Tensor: A list of predicted labels with the same N as the data, the
        accuracy if labels are given and the validation loss if a function is present.
    """
    # Set the model in eval mode and speed up computation time bypassing grad comp.
    self.eval()

    with torch.no_grad():
      # Store intermediary values
      y_out = []
      loss_out = []
      accuracy: typing.List | float = []

      # Do a single pass
      for sample in loader:
        # Send mini-batch data to GPU for faster processing
        X, y = (sample if with_labels else (sample, None))
        X = X.to(self.device)

        # Feedforward through the model
        logits: torch.Tensor = self.forward(X)

        # Compute the current model outputs
        y_hat: torch.Tensor = torch.argmax(logits, dim=1) + 1
        y_out.extend(y_hat.detach().cpu().tolist())

        # Use labels if possible to provide more additional insights
        if not with_labels:
          continue

        # Save matching values
        accuracy.extend((y_hat.detach().cpu() == y + 1).tolist())
        loss_out.extend(loss_fn(logits.detach().cpu(), y).tolist())

    if not with_labels:
      return torch.tensor(y_out, dtype=torch.int)

    if not loss_fn:
      return torch.tensor(y_out, dtype=torch.int), \
             torch.tensor(accuracy, dtype=torch.float).mean().item()

    return torch.tensor(y_out, dtype=torch.int), \
           torch.tensor(accuracy, dtype=torch.float).mean().item(), \
           torch.tensor(loss_out, dtype=torch.float).mean().item()

  def fit(self, train_loader: data.DataLoader, val_loader: data.DataLoader | None,
          loss_fn: nn.Module, optim: Optimizer, n_epochs: int, scheduler: Scheduler | None = None) -> typing.Dict[str, torch.Tensor]:
    """Train the model on the first given loader and optionally evaluate at each epoch
    the metric on the validation loader.

    Args:
        train_loader (data.DataLoader): The loader for the training subset.
        val_loader (data.DataLoader | None): The loader for the validation subset.
        loss_fn (nn.Module): A loss function used for optimization and as a metric.
        optim (Optimizer): The optimizer which changes internally the weights of the NN.
        n_epochs (int): How much the network should train for.
        scheduler (Scheduler | None, optional): Adjust the LR based on loss criteria. Defaults to None.

    Returns:
        typing.Dict[str, torch.Tensor]: Dict containing keys for mode + metric
        and the associated values.
    """
    # Prepare the model for learning
    self.requires_grad_(requires_grad=True)
    self.train()

    # Save global stats
    loss = []
    accy = []
    vccy = []
    vlos = []

    # Do multiple passes in order to converge
    for epoch_i in range(n_epochs):
      epoch_loss: typing.List | float = []
      epoch_accy: typing.List | float = []

      # Do one training pass
      for (X, y) in train_loader:
        # Make one optimization step
        batch_loss, \
        batch_y_hat = self.train_step(X, y, loss_fn, optim)

        # Save intermediary stats
        epoch_loss.append(batch_loss)
        epoch_accy.extend((batch_y_hat == y + 1).tolist())

      # Do one validation pass
      if val_loader is not None:
        _, val_accy, val_loss = self.predict(val_loader, True, loss_fn)
        vccy.append(val_accy)
        vlos.append(val_loss)

      # Compute epoch stats
      epoch_loss = torch.tensor(epoch_loss, dtype=torch.float).mean().item()
      epoch_accy = torch.tensor(epoch_accy, dtype=torch.float).mean().item()

      if self.verbose:
        print('[epoch: {: >4d} / {: <4d}]'.format(epoch_i + 1, n_epochs), end=' ')
        print('[epoch_loss: {:.3f}]'.format(epoch_loss), end=' ')
        print('[epoch_accuracy: {:.3f}]'.format(epoch_accy), end=' ')
        if val_loader is not None:
          print('[valid_loss: {:.3f}]'.format(val_loss), end=' ')
          print('[valid_accuracy: {:.3f}]'.format(val_accy), end=' ')
        print()

      # Save them to history
      loss.append(epoch_loss)
      accy.append(epoch_accy)

      # Adjust LR after each pass
      if scheduler is not None:
        scheduler.step() # val_loss if val_loader else epoch_loss)

    # Return the history of the training and validation processes
    if val_loader is not None:
      return {
        'train_loss': torch.tensor(loss),
        'valid_loss': torch.tensor(vlos),
        'train_accuracy': torch.tensor(accy),
        'valid_accuracy': torch.tensor(vccy),
      }
    else:
      return {
        'train_loss': torch.tensor(loss),
        'train_accuracy': torch.tensor(accy),
      }

  def train_step(self, X: torch.Tensor, y: torch.Tensor,
                 loss_fn: nn.Module, optim: Optimizer) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Do a training step over a mini-batch using the provided data, loss and optimizer.

    Args:
        X (torch.Tensor): A mini-batch of input data.
        y (torch.Tensor): A mini-batch of input labels.
        loss_fn (nn.Module): The loss function to minimize, must not reduce the output.
        optim (Optimizer): The optimizer used to adjust the weights.

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: The loss for the mini-batch
        followed by the predicted list of labels for the given mini-batch.
    """
    # Prepare the model for learning
    self.requires_grad_(requires_grad=True)
    self.train()

    # Send mini-batch data to GPU for faster processing
    X, y = X.to(self.device), y.to(self.device)

    # Feedforward through the model and compute the gradients w.r.t. the weights
    # The CrossEntropyLoss docs say that it accepts logits!
    optim.zero_grad()
    logits: torch.Tensor = self.forward(X)
    loss: torch.Tensor = loss_fn(logits, y).mean()
    loss.backward()
    optim.step()

    # Compute the current model outputs
    y_hat: torch.Tensor = torch.argmax(logits, dim=1) + 1 # [0, 19] -> [1, 20]
    return loss.detach().cpu().item(), y_hat.detach().cpu()


class CompressBlock(nn.Module):
  """Compress the data by pooling along the temporal axis using convolutions."""
  def __init__(self, in_chan: int, out_chan: int, activ_fn: nn.Module,
               bias: bool = True, dropout: float = 0.3) -> None:
    super().__init__()

    # Internal params
    kernel = 3

    # Layers - Apply a convolution then reduce the output and expand the n. of channels
    self.conv = nn.Conv2d(in_chan, out_chan, (1, kernel), 1, 'same', bias=bias)
    self.bn = nn.BatchNorm2d(out_chan)
    self.activ_fn = activ_fn
    self.drop = nn.Dropout2d(p=dropout)
    self.pool = nn.Sequential(
      nn.Conv2d(out_chan, out_chan, (1, 3), stride=(1, 2), padding=(0, 1), bias=bias),
      nn.BatchNorm2d(out_chan),
      activ_fn,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = self.bn(x)
    x = self.activ_fn(x)
    x = self.drop(x)
    x = self.pool(x)
    return x


class DecompressBlock(nn.Module):
  """Decompress the data by unpooling along the temporal axis using transposed
  convolutions and add the necessary padding to obtain the original size."""
  def __init__(self, in_chan: int, out_chan: int, activ_fn: nn.Module,
               bias: bool = True, dropout: float = 0.3, pad_side: int = 0,
               norm: bool = True, unpool_activ_fn: nn.Module | None = None) -> None:
    super().__init__()

    # Internal params
    kernel = 3
    unpool_activ_fn = activ_fn if unpool_activ_fn is None else unpool_activ_fn

    # Layers - Increase the temporal axis by using a TConv then reduce the number
    # of channels
    self.unpool = nn.Sequential(
      nn.ConvTranspose2d(in_chan, in_chan, (1, kernel), (1, 2), padding=(0, 1),
                         output_padding=(0, pad_side), bias=bias),
      nn.BatchNorm2d(in_chan),
      unpool_activ_fn,
    )
    self.conv = nn.Conv2d(in_chan, out_chan, (1, kernel), 1, padding='same', bias=bias)
    self.bn = nn.BatchNorm2d(out_chan) if norm else nn.Identity()
    self.activ_fn = activ_fn
    self.drop = nn.Dropout2d(p=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.unpool(x)
    x = self.conv(x)
    x = self.bn(x)
    x = self.activ_fn(x)
    x = self.drop(x)
    return x


class Encoder(nn.Module):
  """Apply a number of compression blocks to reduce the initial representation
  then squeeze the spatial and temporal dimensions into a fixed embedding size
  for the bottleneck."""
  def __init__(self, in_chan: int, filter_chan: int, embedding_features: int,
               activ_fn: nn.Module, bias: bool = True, dropout: float = 0.3) -> None:
    super().__init__()

    # Layers - Three compression blocks which reduce the temporal axis and
    # expand the n. of channels. Apply a linear layer at the end (using conv).
    self.comp1 = CompressBlock(in_chan, filter_chan, activ_fn, bias, dropout)
    self.comp2 = CompressBlock(filter_chan, filter_chan * 2, activ_fn, bias, dropout)
    self.comp3 = CompressBlock(filter_chan * 2, filter_chan * 4, activ_fn, bias, dropout)
    self.embed = nn.Sequential(
      nn.Conv2d(filter_chan * 4, embedding_features, (1, 19), bias=bias),
      nn.BatchNorm2d(embedding_features),
      activ_fn,
      nn.Flatten(),
      nn.Dropout(p=dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.comp1(x)
    x = self.comp2(x)
    x = self.comp3(x)
    x = self.embed(x)
    return x


class Decoder(nn.Module):
  """Recover the spatial and temporal axis from the bottleneck by applying
  a TConv. Then reduce the n. of channels and expand the temporal axis to the
  original input size."""
  def __init__(self, embedding_features: int, filter_chan: int, out_chan: int,
               activ_fn: nn.Module, bias: bool = True, dropout: float = 0.3) -> None:
    super().__init__()

    # Layers - Apply a TConv and decompress the embeddings to the original size.
    self.unembed = nn.Sequential(
      nn.Unflatten(dim=1, unflattened_size=(embedding_features, 1, 1)),
      nn.ConvTranspose2d(embedding_features, filter_chan * 4, (1, 19), bias=bias),
      nn.BatchNorm2d(filter_chan * 4),
      activ_fn,
      nn.Dropout2d(p=dropout),
    )
    self.decomp1 = DecompressBlock(filter_chan * 4, filter_chan * 2, activ_fn, bias, dropout, 1)
    self.decomp2 = DecompressBlock(filter_chan * 2, filter_chan, activ_fn, bias, dropout, 0)
    self.decomp3 = DecompressBlock(filter_chan, out_chan, nn.Identity(), bias, 0, 1, False,
                                   unpool_activ_fn=activ_fn)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.unembed(x)
    x = self.decomp1(x)
    x = self.decomp2(x)
    x = self.decomp3(x)
    return x


class AutoEncoder(NNModule):
  """Compresses the input data to a lower dimensional, latent representation,
  using an Encoder and reconstruct the information by applying a Decoder.
  The bottleneck should represent a smaller feature vector than the original
  data, which reflects the essential characteristics of the present information."""
  def __init__(self, in_chan: int, filter_chan: int, embedding_features: int,
               activ_fn: nn.Module, dropout: float = 0.3, bias: bool = True,
               init_fn: typing.Callable[[torch.Tensor], None] | None = None,
               device: torch.device | None = None, verbose: bool = False) -> None:
    super().__init__(init_fn=init_fn, device=device, verbose=verbose, bias=bias)

    # Layers - Data -> Encoder -> Bottleneck -> Decoder -> (Data' ~ Data)
    self.enc = Encoder(in_chan, filter_chan, embedding_features, activ_fn, bias, dropout)
    self.dec = Decoder(embedding_features, filter_chan, in_chan, activ_fn, bias, dropout)

    # Apply initialization
    if self.init_fn:
      self.apply(self._weights_initialization)

  def encode(self, x: torch.Tensor | data.DataLoader) -> torch.Tensor:
    """Encodes inputs of size (N, 3, 1, 150) to (N, embedding_features) for
    a lower-dimensional representation. Can process either a mini-batch or
    an entire data.DataLoader."""
    if isinstance(x, torch.Tensor):
      return self.enc(x)

  def decode(self, x: torch.Tensor | data.DataLoader) -> torch.Tensor:
    """Expects emeddings of size (N, embedding_features) and decodes them
    to (N, 3, 1, 150). Can process either a mini-batch or
    an entire data.DataLoader."""
    return self.dec(x)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Encode and decode in one step - the full pipeline
    return self.decode(self.encode(x))

  def predict(self, loader: data.DataLoader, with_labels: bool = True,
              loss_fn: nn.Module | None = None) -> torch.Tensor | typing.Tuple[torch.Tensor, torch.Tensor]:
    """Given a DataLoader, encode the data and expose a bottleneck as the output.

    Args:
        loader (data.DataLoader): The loader serving data to be predicted on.
        with_labels (bool, optional): Ignored. Defaults to True.
        loss_fn (nn.Module | None, optional): Used to optionally compute a loss. Defaults to None.

    Returns:
        torch.Tensor | typing.Tuple[torch.Tensor, torch.Tensor]: Return the predicted
        embeddings over the given data and the loss if loss_fn is given.
    """
    self.eval()
    with torch.no_grad():
      # Store validation stats
      vl_loss = torch.zeros(len(loader))
      embeddings = []

      # Do prediction pass
      for batch_i, sample in enumerate(loader):
        # Extract the data from the mini-batch
        X = sample if with_labels is False else sample[0]

        # Send the input data to the GPU
        X: torch.Tensor = X.to(self.device)

        # Compute the output and restoration loss
        bottleneck = self.encode(X)
        embeddings.append(bottleneck.cpu())

        # Also, compute loss if loss_fn is given
        if loss_fn is None:
          continue

        # Restore the initial input
        XR = self.decode(bottleneck)
        restoration_loss: torch.Tensor = loss_fn(XR, X).mean()

        # Store mini-batch loss
        vl_loss[batch_i] = restoration_loss.cpu().item()

      # Transform embeddings into proper tensor format
      embeddings = torch.vstack(embeddings)

      # Return only the embeddings as no loss could be computed
      if not loss_fn:
        return embeddings

      # Return the embeddings and the associated mean loss
      return embeddings, vl_loss.mean().item()

  def fit(self, train_loader: data.DataLoader, val_loader: data.DataLoader | None,
                loss_fn: nn.Module, optim: Optimizer, n_epochs: int, scheduler: Scheduler | None = None) -> typing.Dict[str, torch.Tensor]:
    """Train the model to reconstruct the given data. Evaluates the reconstruction
    on the validation subset if given.

    Args:
        train_loader (data.DataLoader): The loader over the training data subset.
        val_loader (data.DataLoader | None): The loader over the optional validation subset.
        loss_fn (nn.Module): The loss function used to train the model.
        optim (Optimizer): The optimizer which minimizes the reconstruction loss.
        n_epochs (int): A number of phases that the model should train for.
        scheduler (Scheduler | None, optional): Specifies how the optimizer should
        change the LR to smoothen out the training process. Defaults to None.

    Returns:
        typing.Dict[str, torch.Tensor]: Dict containing keys for mode + metric
        and the associated values.
    """
    # Make sure the model is setup for training
    self.requires_grad_(True)

    # Store history for the entire training process
    tr_loss = torch.zeros(n_epochs)
    vl_loss = torch.zeros(n_epochs)

    # Train the model for a number of specified epochs
    for epoch_i in range(n_epochs):
      # Make sure the model is setup for training
      self.train()

      # Use epoch stats for progress tracking
      e_tr_loss = torch.zeros(len(train_loader), device=self.device)
      if val_loader is not None:
        e_vl_loss = torch.zeros(len(val_loader), device=self.device)

      # Make a training step over the data
      for batch_i, (X, y) in enumerate(train_loader):
        # Move the data to the GPU for faster processing on the given data
        X: torch.Tensor = X.to(self.device)

        # Reset current state
        optim.zero_grad()

        # Do forward pass across the mini-batch
        bottleneck: torch.Tensor = self.encode(X)
        XR: torch.Tensor = self.decode(bottleneck)

        # Compute loss & propagate & minimize restoration loss
        restoration_loss: torch.Tensor = loss_fn(XR, X).mean()
        restoration_loss.backward()
        optim.step()

        # Store train stats
        e_tr_loss[batch_i] = restoration_loss.detach().item()

      # Store the losses for progress tracking
      tr_loss[epoch_i] = e_tr_loss.detach().mean().cpu().item()

      # Make a validation pass at the end of the epoch
      if val_loader is not None:
        self.eval()
        with torch.no_grad():
          for batch_i, (X, _) in enumerate(val_loader):
            # Send the input data to the GPU
            X: torch.Tensor = X.to(self.device)

            # Compute the output and restoration loss
            XR = self.forward(X)
            restoration_loss: torch.Tensor = loss_fn(XR, X).mean()

            # Store mini-batch loss
            e_vl_loss[batch_i] = restoration_loss.item()

        # Store the losses for progress tracking
        vl_loss[epoch_i] = e_vl_loss.mean().cpu().item()

      # Adjust LR after each pass
      if scheduler is not None:
        scheduler.step(vl_loss[epoch_i] if val_loader else tr_loss[epoch_i])

      # Display current performance
      if self.verbose:
        print('[epoch: {: >4d} / {: <4d}]'.format(epoch_i + 1, n_epochs), end=' ')
        print('[epoch_loss: {:.3f}]'.format(tr_loss[epoch_i]), end=' ')
        if val_loader is not None:
          print('[valid_loss: {:.3f}]'.format(vl_loss[epoch_i]), end=' ')
        print()

    # Return the current stats
    if val_loader is not None:
      return { 'train_loss': tr_loss, 'valid_loss': vl_loss }
    else:
      return { 'train_loss': tr_loss }

