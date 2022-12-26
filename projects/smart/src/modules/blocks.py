import torch
import typing
import numpy as ny
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as pt
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.optim.optimizer import Optimizer
from collections import OrderedDict


class Swish(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RepeatModule(nn.Module):
  def __init__(self, module_gen: typing.Callable[[], nn.Module], times: int):
    super(RepeatModule, self).__init__()
    self.layers_ = nn.Sequential()

    # Generate N modules and chain them sequentially
    for i in range(times):
      self.layers_.add_module(name=f'module_{i}', module=module_gen())

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers_(x)


class ResModule1d(nn.Module):
  def __init__(self, in_chan: int, out_chan: int, module: nn.Module = None, bottleneck: int = 1, bias: bool = True):
    super(ResModule1d, self).__init__()
    self.layers_ = nn.Sequential(module)
    self.bottleneck_ = bottleneck
    self.in_chan = in_chan
    self.out_chan = out_chan

    if bottleneck != 1:
      self.layers_.insert(0, nn.Conv1d(in_chan, in_chan // bottleneck, 1, bias=bias))
    if bottleneck != 1 or in_chan != out_chan:
      self.layers_.append(nn.Conv1d(in_chan // bottleneck, out_chan, 1, bias=bias))
    if in_chan != out_chan:
      self.res_ = nn.Conv1d(in_chan, out_chan, 1, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.in_chan == self.out_chan:
      return x + self.layers_(x)
    else:
      return self.res_(x) + self.layers_(x)


class ResBlock1d(nn.Module):
  def __init__(self, in_chan: int, out_chan: int, kernel_size: int,
               activ_fn: nn.Module = nn.ReLU, norm: bool = True,
               bottleneck: int = 1, times: int = 3, bias: bool = True,
               dropout: float = 0.0):
    super(ResBlock1d, self).__init__()

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
    self.layers_ = ResModule1d(in_chan, out_chan, module, bottleneck)

  def __create_block(self) -> nn.Module:
    layers = nn.Sequential()

    layers.append(nn.Conv1d(self.in_chan // self.bottleneck,
                            self.in_chan // self.bottleneck,
                            self.kernel_size,
                            padding='same',
                            bias=self.bias))

    if self.norm is True:
      layers.append(nn.BatchNorm1d(self.in_chan // self.bottleneck))

    if self.activ_fn is not None:
      layers.append(self.activ_fn())

    if self.dropout is not None:
      layers.append(nn.Dropout1d(p=self.dropout))

    return layers

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers_(x)


class SelfAttentionBlock1d(nn.Module):
  def __init__(self, in_chan: int, bottleneck: int = 1, bias: bool = True) -> None:
    super().__init__()
    embed_chan = in_chan // bottleneck

    # Create single-attention layers
    self.conv_query = nn.Conv1d(in_chan, embed_chan, 1, bias=bias)
    self.conv_key = nn.Conv1d(in_chan, embed_chan, 1, bias=bias)
    self.conv_value = nn.Conv1d(in_chan, embed_chan, 1, bias=bias)
    self.conv_out = nn.Conv1d(embed_chan, in_chan, 1, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Embed input using bottleneck
    query = self.conv_query(x)
    key = self.conv_key(x)
    value = self.conv_value(x)

    # Compute attention map
    query_key = query.transpose(1, 2) @ key
    attention = torch.softmax(query_key, dim=2)

    # Apply attention over the input
    output = torch.transpose(attention @ value.transpose(1, 2), 1, 2)

    # Add residual connection
    return x + self.conv_out(output)


class MultiHeadAttentionBlock1d(nn.Module):
  def __init__(self, in_chan: int, heads: int = 4, bias: bool = True, dropout: float = 0.0) -> None:
    super().__init__()

    self.attention_ = nn.MultiheadAttention(embed_dim=in_chan, num_heads=heads,
                                            bias=bias, batch_first=True,
                                            dropout=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply MHAttention over mini-batch of size: (N, C, T)"""
    x = x.transpose(1, 2)
    attn, _ = self.attention_.forward(x, x, x)
    out = attn.transpose(1, 2)
    return out


class GlobalMaxPool(nn.Module):
  def __init__(self, dim: int = 2) -> None:
    super().__init__()
    self.dim = dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, dim=self.dim).values


class TemporalAttentionNN(nn.Module):
  def __init__(self, in_chan: int, out_chan: int,
               s_filters: int = 3, n_filters: int = 128, n_units: int = 1024,
               activ_fn: nn.Module = Swish, norm: bool = True,
               bottleneck: int = 2, dropout: float = 0.3, bias: bool = True,
               init_fn: typing.Callable[[torch.Tensor], None] = None,
               device: torch.device = None, verbose: bool = False):
    super().__init__()
    self._device = device
    self.verbose = verbose
    self.init_fn = init_fn
    self.inner_repeat = 3
    self.bias = bias

    # --- Input Layer ---
    self.layers_ = nn.Sequential()
    self.layers_.append(nn.Conv1d(in_chan, n_filters, s_filters, bias=bias, padding='same'))

    # --- Stage 1 Residual Blocks ---
    self.layers_.add_module(name='stage_1', module=nn.Sequential(
      ResBlock1d(n_filters, n_filters, s_filters, activ_fn, norm,
                 bottleneck, bias, self.inner_repeat, dropout),
      SelfAttentionBlock1d(n_filters, bottleneck, bias),
      nn.MaxPool1d(s_filters, 2),
    ))

    # --- Stage 2 Residual Blocks ---
    self.layers_.add_module(name='stage_2', module=nn.Sequential(
      ResBlock1d(n_filters, n_filters * 2, s_filters, activ_fn, norm,
                 bottleneck, bias, self.inner_repeat, dropout),
      ResBlock1d(n_filters * 2, n_filters * 2, s_filters, activ_fn, norm,
                 bottleneck, bias, self.inner_repeat, dropout),
      SelfAttentionBlock1d(n_filters * 2, bottleneck, bias),
      nn.MaxPool1d(s_filters, 2),
    ))

    # --- Stage 3 Residual Blocks ---
    self.layers_.add_module(name='stage_3', module=nn.Sequential(
      ResBlock1d(n_filters * 2, n_filters * 4, s_filters, activ_fn, norm,
                 bottleneck, bias, self.inner_repeat, dropout),
      ResBlock1d(n_filters * 4, n_filters * 4, s_filters, activ_fn, norm,
                 bottleneck, bias, self.inner_repeat, dropout),
      ResBlock1d(n_filters * 4, n_filters * 4, s_filters, activ_fn, norm,
                 bottleneck, bias, self.inner_repeat, dropout),
      SelfAttentionBlock1d(n_filters * 4, bottleneck, bias),
      nn.MaxPool1d(s_filters, 2),
    ))

    # --- Global Pooling ---
    self.layers_.append(GlobalMaxPool(dim=2))

    # --- First FCN Layer ---
    self.layers_.add_module(name='fcn_1', module=nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(n_filters * 4, n_units * 2, bias),
      nn.BatchNorm1d(n_units * 2) if norm else nn.Identity(),
      activ_fn(),
    ))

    # --- Intermediary FCN Layers ---
    self.layers_.add_module(name=f'fcn_2', module=nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(n_units * 2, n_units * 2, bias),
      nn.BatchNorm1d(n_units * 2) if norm else nn.Identity(),
      activ_fn(),
    ))
    self.layers_.add_module(name=f'fcn_3', module=nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(n_units * 2, n_units * 2, bias),
      nn.BatchNorm1d(n_units * 2) if norm else nn.Identity(),
      activ_fn(),
    ))

    # --- Final Layer ---
    self.layers_.append(nn.Linear(n_units * 2, out_chan, bias))

    # Optionally initialize the weights with a custom approach
    if self.init_fn:
      self.apply(self.__weights_initialization)

  def __weights_initialization(self, module: nn.Module) -> None:
    # Define range of modules that will be weight-initialized according to init_fn
    allowed_types = tuple([
      nn.Linear,
      nn.Conv1d,
    ])

    # Apply the init_fn
    if isinstance(module, allowed_types):
      self.init_fn(module.weight)

      if self.bias:
        torch.nn.init.zeros_(module.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers_(x)

  def predict(self, loader: data.DataLoader, with_labels: bool = True) -> typing.Tuple[torch.Tensor, ...]:
    """Predict the list of labels for the given data (not a mini-batch). Do not SHUFFLE!

    Args:
        data (torch.Tensor): The list of input data.
        with_labels (bool): Specifies if labels are served in the loader (ex. for validation).

    Returns:
        torch.Tensor: A list of predicted labels with the same N as the data and
        accuracy if labels are given.
    """
    # Set the model in eval mode and speed up computation time bypassing grad comp.
    self.eval()

    with torch.no_grad():
      # Store intermediary values
      y_out = []
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

    if with_labels:
      return torch.tensor(y_out, dtype=torch.int), \
             torch.tensor(accuracy, dtype=torch.float).mean().item()
    else:
      return torch.tensor(y_out, dtype=torch.int)

  def fit(self, train_loader: data.DataLoader, val_loader: data.DataLoader | None,
            loss_fn: nn.Module, optim: Optimizer, n_epochs: int, scheduler: Scheduler | None = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Prepare the model for learning
    self.requires_grad_(requires_grad=True)
    self.train()

    # Save global stats
    loss = []
    accy = []
    vccy = []

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
        _, val_accy = self.predict(val_loader, with_labels=True)
        vccy.append(val_accy)

      # Compute epoch stats
      epoch_loss = torch.tensor(epoch_loss, dtype=torch.float).mean().item()
      epoch_accy = torch.tensor(epoch_accy, dtype=torch.float).mean().item()

      if self.verbose:
        print('[epoch: {: >4d} / {: <4d}]'.format(epoch_i + 1, n_epochs), end=' ')
        print('[epoch_loss: {:.3f}]'.format(epoch_loss), end=' ')
        print('[epoch_accuracy: {:.3f}]'.format(epoch_accy), end=' ')
        if val_loader is not None:
          print('[valid_accuracy: {:.3f}]'.format(val_accy), end=' ')
        print()

      # Save them to history
      loss.append(epoch_loss)
      accy.append(epoch_accy)

      # Adjust LR after each pass
      if scheduler is not None:
        scheduler.step()

    # Return the history of the training and validation processes
    if val_loader is not None:
      return torch.tensor(loss), torch.tensor(accy), \
            (torch.tensor(vccy) if val_loader is not None else None)
    else:
      return torch.tensor(loss), torch.tensor(accy)

  def train_step(self, X: torch.Tensor, y: torch.Tensor,
                 loss_fn: nn.Module, optim: Optimizer) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Do a training step over a mini-batch using the provided data, loss and optimizer.

    Args:
        X (torch.Tensor): A mini-batch of input data.
        y (torch.Tensor): A mini-batch of input labels.
        loss_fn (nn.Module): The loss function to minimize.
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
    loss: torch.Tensor = loss_fn(logits, y)
    loss.backward()
    optim.step()

    # Compute the current model outputs
    y_hat: torch.Tensor = torch.argmax(logits, dim=1) + 1 # [0, 19] -> [1, 20]
    return loss.detach().cpu().item(), y_hat.detach().cpu()

  @property
  def device(self) -> torch.device:
    if self._device is None:
      return torch.device('cpu')
    else:
      return self._device


class SmartDataset(data.Dataset):
  def __init__(self, data: ny.ndarray, labels: ny.ndarray | None = None,
               n_chans: int = 3, n_features: int = 150) -> None:
    super().__init__()

    # Save given input data
    self.data_ = torch.tensor(data).reshape((-1, n_chans, n_features))
    self.labels_ = None if labels is None else torch.tensor(labels) - 1

  @property
  def data(self) -> torch.Tensor:
    return self.data_

  @property
  def labels(self) -> torch.Tensor | None:
    return None if self.labels_ is None else self.labels_

  def __getitem__(self, index: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor | None]:
    # Select the desired sample and keep other dimensions to allow slicing
    x_sample = self.data_[index, ...]

    # Return both the sample and its label
    if self.labels_ is not None:
      return x_sample, self.labels_[index, ...]

    # Or only the sample if the label is missing
    return x_sample

  def __len__(self) -> int:
    return self.data_.shape[0]
