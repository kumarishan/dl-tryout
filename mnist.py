from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from datasets import Dataset, Datasets


class MNISTDataset(Dataset):
  def __init__(self, dataset):
    self._mnist = dataset
    self._n_examples = dataset.num_examples
    self._x_shape = [28, 28]
    self._y_shape = [1]
    super(MNISTDataset, self).__init__()

  def _load_dataset(self):
    return

  def next_batch(self, batch, batch_size):
    """Next batch
    Args:
      batch: Int. Batch index
      batch_size: Int. Size of batch
    Returns:
      Images and labels
    """
    start = batch * batch_size
    end = (batch + 1) * batch_size
    x = self._mnist.images[start:end]
    y = self._mnist.labels[start:end]
    x = np.reshape(x, [batch_size, 28, 28])
    y = np.reshape(y, [batch_size, 1])
    return x, y


class MNISTDatasets(Datasets):
  """MNIST datasets with train, test and validation set"""

  def __init__(self, data_dir):
    """Create MNIST datasets
    Args:
      data_dir: String. Directory to store data files
    """
    self.data_dir = data_dir
    datasets = self._load_dataset()
    self._datasets = datasets
    train = MNISTDataset(datasets.train)
    test = MNISTDataset(datasets.test)
    validation = MNISTDataset(datasets.validation)
    super(MNISTDatasets, self).__init__(train=train, test=test, validation=validation)

  def _load_dataset(self):
    """Load the dataset"""
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    return input_data.read_data_sets(self.data_dir, one_hot=False)

  def norms(self):
    """Calculate mean and std
    Returns:
      mean and std
    """
    mean = np.reshape(np.mean(self._datasets.train.images, axis=0), [28, 28])
    std = np.reshape(np.std(self._datasets.train.images, axis=0), [28, 28])
    return mean, std


if __name__ == '__main__':
  mnist = MNISTDatasets('data/MNIST')
  mean, std = mnist.norms()
  print('Mean = %s\nStd = %s' % (mean, std))
  n_examples = mnist.train.n_examples
  batch_size = 500
  for batch in trange(n_examples // batch_size):
    x, y = mnist.train.next_batch(batch, batch_size)
    tqdm.write('Train Batch %d: x = %s, y = %s' % (batch, x.shape, y.shape))
    time.sleep(0.5)