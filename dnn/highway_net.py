from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from datasets.cifar10 import CIFAR10Datasets

Highway = namedtuple('Highway', 'n_highways kernel_sz')
Stride = namedtuple('Stride', 'highway out_ch kernel_sz stride n_repeat')

class HighwayNet:
  def __init__(self, x_shape, n_classes, strides):
    self.x_shape = x_shape
    self.n_classes = n_classes
    self.strides = strides

    self._build()

  def _highway(self, x, kernel_sz, stride_i, layer_i):
    with tf.variable_scope('highway_unit_%d_%d' % (stride_i, layer_i)):
      out_ch = int(x.get_shape()[3])
      H = slim.conv2d(x, out_ch, kernel_sz)
      T = slim.conv2d(x, out_ch, kernel_sz, biases_initializer=tf.constant_initializer(-1.0),
                      activation_fn=tf.nn.sigmoid)
      h = H * T + x * (1.0 - T)
    return h

  def _create_highway_network(self, x):
    with tf.name_scope('highway_network'):
      curr_in = x
      i = 0
      for _, stride in enumerate(self.strides):
        for _ in range(stride.n_repeat):
          highway = stride.highway
          if highway is not None:
            n_highways = highway.n_highways
            with tf.name_scope('highway'):
              for j in range(n_highways):
                curr_in = self._highway(curr_in, highway.kernel_sz, i, j)

          curr_in = slim.conv2d(curr_in, stride.out_ch, stride.kernel_sz, stride=stride.stride,
                                normalizer_fn=slim.batch_norm, scope='stride_%d' % i)
          i += 1

      # TODO define this layer
      conv_h = slim.conv2d(curr_in, self.n_classes, [3, 3], normalizer_fn=slim.batch_norm, activation_fn=None,
                           scope='conv_h')

    with tf.name_scope('softmax'):
      y_pred = slim.layers.softmax(slim.layers.flatten(conv_h))

    return y_pred

  def _create_loss_layer(self, y_oh, y_pred):
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(-tf.reduce_sum(y_oh * tf.log(y_pred) + 1e-10, reduction_indices=[1]))
    return loss

  def _build(self):
    self.x = tf.placeholder(tf.float32, shape=([None] + self.x_shape), name='x')
    self.y = tf.placeholder(tf.int32, shape=[None])
    self.y_oh = slim.layers.one_hot_encoding(self.y, self.n_classes)
    self.y_pred = self._create_highway_network(self.x)
    self.loss = self._create_loss_layer(self.y_oh, self.y_pred)

  def fit(self, datasets, learning_rate=0.001, n_epochs=50, batch_size=100, norm=False, model_dir='./'):
    """Fit the dataset
    Args:
      datasets: Datasets. Dataset to train.
      learning_rate: float. Optional learning rate.
      n_epochs: int. Optional number of epochs.
      norm: boolean. Optional normalize x (input)
    """
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    x_mean, x_std = datasets.norms()
    minimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      for epoch in trange(n_epochs, desc='Epoch'):
        epoch_accuracy = 0.
        epoch_loss = 0.
        for batch in trange(datasets.train.n_examples // batch_size, desc='Batch'):
          x_batch, y_batch = datasets.train.next_batch(batch, batch_size)
          y_batch = np.squeeze(y_batch)
          x_batch = x_batch / 256.0
          x_batch = (x_batch - x_mean) / x_std
          _, loss, y_pred = sess.run([minimize, self.loss, self.y_pred],
                                     feed_dict={self.x: x_batch, self.y: y_batch})
          epoch_accuracy += np.sum(np.equal(y_batch, np.argmax(y_pred, 1)))
          epoch_loss += loss
        epoch_accuracy /= datasets.train.n_examples
        epoch_loss /= datasets.train.n_examples

        tqdm.write('Epoch %d: Accuracy = %.2f Loss = %.2f' % (epoch, epoch_accuracy, epoch_loss))


if __name__ == '__main__':
  cifar10 = CIFAR10Datasets('data/cifar10')
  strides = [
    Stride(highway=None, out_ch=32, kernel_sz=[3, 3], stride=[1, 1], n_repeat=1),
    Stride(highway=Highway(n_highways=2, kernel_sz=[3, 3]),
           out_ch=32, kernel_sz=[3, 3], stride=[2, 2], n_repeat=5),
  ]
  highway_net = HighwayNet([32, 32, 3], 10, strides=strides)
  highway_net.fit(cifar10)