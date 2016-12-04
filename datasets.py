from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


class Datasets(object):
  def __init__(self, train=None, test=None, validation=None):
    super(Datasets, self).__init__()
    self._train = train
    self._test = test
    self._validation = validation

  @property
  def train(self):
    return self._train

  @property
  def test(self):
    return self._test

  @property
  def validation(self):
    return self._validation


class Dataset(object):
  def __init__(self):
    super(Dataset, self).__init__()
    self._load_dataset()

  def _load_dataset(self):
    raise NotImplementedError()

  @property
  def x_shape(self):
    return self._x_shape

  @property
  def y_shape(self):
    return self._y_shape

  @property
  def n_examples(self):
    return self._n_examples

  def next_batch(self, batch, batch_size):
    raise NotImplementedError()