from __future__ import division
from __future__ import print_function

import math
import os.path

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from datasets import Dataset, Datasets


def unpickle(file):
  import cPickle
  with open(file, 'rb') as fo:
    dict = cPickle.load(fo)
  return dict

class CIFAR10TrainDataset(Dataset):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self._n_examples = 50000
    self._x_shape = [32, 32, 3]
    self._y_shape = [1]
    super(CIFAR10TrainDataset, self).__init__()

  def next_batch(self, batch, batch_size):
    """Get next batch
    Args:
      batch: int. Index of the batch starting with 0
      batch_size: int. Batch size
    Returns
      np.ndarray batch of data
    """
    n_batches_per_file = 10000 // batch_size
    file_i = batch // n_batches_per_file + 1
    if file_i != self._file_i:
      self._load_file(file_i)

    curr_batch = batch % n_batches_per_file
    x = self._cifar['data'][curr_batch * batch_size:(curr_batch + 1) * batch_size]
    x = np.reshape(x, [batch_size, 32, 32, 3], order='F')
    y = np.hstack(self._cifar['labels'][curr_batch * batch_size:(curr_batch + 1) * batch_size])
    y = np.reshape(y, [batch_size, 1])
    return x, y

  def _load_dataset(self):
    self._load_file(1)

  def _load_file(self, file_i):
    self._file_i = file_i
    self._cifar = unpickle(self._file_name(file_i))

  def _file_name(self, file_i):
    return os.path.join(self.data_dir, 'data_batch_%d' % file_i)


class CIFAR10TestDataset(Dataset):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self._n_examples = 10000
    self._x_shape = [32, 32, 3]
    self._y_shape = [1]
    super(CIFAR10TestDataset, self).__init__()

  def _load_dataset(self):
    """Loads the dataset"""
    self._cifar = unpickle(self._data_file)

  @property
  def _data_file(self):
    """Returns data file"""
    return os.path.join(self.data_dir, 'test_batch')

  def next_batch(self, batch, batch_size):
    """Next batch
    Args:
      batch: Int. Batch index (starts from 0)
      batch_size: Int. Size of batch
    Returns
      x and y
    """
    x = self._cifar['data'][batch * batch_size:(batch + 1) * batch_size]
    x = np.reshape(x, [batch_size, 32, 32, 3], order='F')
    y = np.hstack(self._cifar['labels'][batch * batch_size:(batch + 1) * batch_size])
    y = np.reshape(y, [batch_size, 1])
    return x, y


class CIFAR10Datasets(Datasets):
  """CIFAR 10 Datasets"""

  def __init__(self, data_dir):
    """Create CIFAR 10 Datasets
    Args:
      data_dir: String. Path to store data files
    """
    train = CIFAR10TrainDataset(data_dir)
    test = CIFAR10TestDataset(data_dir)
    super(CIFAR10Datasets, self).__init__(train=train, test=test)

  def norms(self):
    """Calculate mean and std
    Returns:
      mean and std
    """
    batch_size = 10000
    mean = np.zeros(self.train.x_shape, dtype=np.float, order='F')
    for batch in range(self.train.n_examples // batch_size):
      x, _ = self.train.next_batch(batch, batch_size)
      mean += np.sum(x, axis=0)
    mean /= (self.train.n_examples * 256.0)

    std = np.zeros(self.train.x_shape, dtype=np.float, order='F')
    for batch in range(self.train.n_examples // batch_size):
      x, _ = self.train.next_batch(batch, batch_size)
      std += np.sum(np.abs(x / 256.0 - mean)**2, axis=0)
    std /= self.train.n_examples

    return mean, std


if __name__ == '__main__':
  cifar10 = CIFAR10Datasets('data/cifar10')
  n_examples = cifar10.train.n_examples
  mean, std = cifar10.norms()
  print('Mean = %s\n Std = %s' % (mean, std))
  batch_size = 1000
  for batch in trange(n_examples // batch_size):
    x, y = cifar10.train.next_batch(batch, batch_size)
    tqdm.write('Batch %d: x = %s, y = %s, file = data_batch_%d' % (batch, x.shape, y.shape, cifar10.train._file_i))