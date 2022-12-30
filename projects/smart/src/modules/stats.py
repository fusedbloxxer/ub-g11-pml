import os as so
import click
import numpy as ny
from os import path
import pandas as ps
import pathlib as pb
from typing import List
import matplotlib.pyplot as pt
import processing as pg
from numpy.random import default_rng
import pandas as pd


@click.group(invoke_without_command=True)
@click.option('--seed', default=87, show_default=True, type=int)
@click.option('--data-folder', default=pb.Path('..', '..', 'data'), show_default=True, type=click.Path(exists=True, dir_okay=True, path_type=pb.Path))
@click.pass_context
def data_analysis_cli(ctx: click.Context, data_folder: pb.Path, seed: int):
  ctx.obj['data_folder'] = data_folder
  ctx.obj['seed'] = seed


@data_analysis_cli.command()
@click.pass_context
def plot_missing_values(ctx: click.Context) -> None:
  # Compute sizes in order to search for potential missing values in each time series
  dataset: pg.Dataset = pg.Dataset(ctx.obj['data_folder'], ctx.obj['seed'])
  train_sizes = [sample.shape[0] for sample in dataset.train_data]
  test_sizes = [sample.shape[0] for sample in dataset.train_data]

  # Display missing value stats
  pt.figure(figsize=(10, 2.5))
  for i, (sizes, title) in enumerate([(train_sizes, 'train'), (test_sizes, 'test')]):
    pt.subplot(1, 2, i + 1)
    pt.title(f'Missing values in {title.capitalize()}')
    pt.xlabel('Time Series Size')
    pt.ylabel('Frequency')
    pt.hist(sizes)
    pt.grid(True)
  pt.show()


@data_analysis_cli.group(name='stats', invoke_without_command=True)
@click.pass_context
def display_train_stats(ctx: click.Context):
  # Show stats from the training subset
  if not ctx.invoked_subcommand:
    dataset: pg.Dataset = pg.Dataset(ctx.obj['data_folder'], ctx.obj['seed'])
    train_subset: pd.DataFrame = dataset.to_pandas()
    click.echo(train_subset.describe())


@display_train_stats.command(name='outliers')
@click.option('--remove', is_flag=True, default=False, show_default=True)
@click.option('--by', default='class', show_default=True, type=click.Choice(['global', 'class'], case_sensitive=False))
@click.option('--iqr', default=1.5, type=float, show_default=True)
@click.pass_context
def display_train_boxplots(ctx: click.Context, remove: bool, by: str, iqr: float):
  # Show boxplots for each class from the training subset
  dataset: pg.Dataset = pg.Dataset(ctx.obj['data_folder'], ctx.obj['seed'])

  # Remove the outliers according to the given options
  if remove:
    dataset.remove_outliers(by=by, factor=iqr)

  # Get the processed data and display it
  train_subset: pd.DataFrame = dataset.to_pandas()
  train_subset.groupby('label')[['x', 'y', 'z']].boxplot(figsize=(15, 15))
  pt.show()


if __name__ == '__main__':
  data_analysis_cli(obj={})

