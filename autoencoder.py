from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


class Autoencoder:
  """Simple Autoencoder"""

  def __init__(self, input_size, hidden_dims=[512, 256, 64], learning_rate=0.001,
               batch_size=50, n_epochs=10):
    self.input_size = input_size
    self.hidden_dims = hidden_dims
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.n_epochs = n_epochs

    self._build()

  def _build(self):
    """Build the network"""
    self.x = tf.placeholder(tf.float32, [None, self.input_size], name='x')
    curr_in = self.x

    with tf.name_scope('encoder'):
      weights = []
      for layer_i, dim_output in enumerate(self.hidden_dims):
        dim_input = int(curr_in.get_shape()[1])
        with tf.name_scope('hidden_%d' % layer_i):
          W = tf.Variable(
                tf.truncated_normal([dim_input, dim_output],
                                    stddev=1.0 / math.sqrt(float(dim_input))),
                                    name='Weights')
          b = tf.Variable(tf.zeros([dim_output]), name='biases')
          weights.append(W)
          h_layer = tf.nn.tanh(tf.matmul(curr_in, W) + b)
          curr_in = h_layer

    self.z = curr_in
    weights.reverse()

    with tf.name_scope('decoder'):
      dims = self.hidden_dims[:-1][::-1] + [self.input_size]
      for layer_i, dim_output in enumerate(dims):
        with tf.name_scope('hidden_%d' % layer_i):
          W = tf.transpose(weights[layer_i])
          b = tf.Variable(tf.zeros([dim_output]), name='biases')
          h_layer = tf.nn.tanh(tf.matmul(curr_in, W) + b)
          curr_in = h_layer

    self.y = curr_in

    self.loss = tf.reduce_sum(tf.square(self.y - self.x))
    self.minimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

  def fit(self, datasets, x_mean=None):
    """Fit dataset"""
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())

      for epoch in trange(self.n_epochs, desc='Epoch'):
        for batch_i in trange(datasets.train.num_examples // self.batch_size, desc='Batch'):
          x_batch, _ = datasets.train.next_batch(self.batch_size)
          if x_mean is not None:
            x_batch = np.array([x - x_mean for x in x_batch])
          _, loss = sess.run([self.minimize, self.loss], feed_dict={self.x: x_batch})

        tqdm.write('\nEpoch %d: Loss = %.2f' % (epoch, loss))

      x_test, _ = datasets.test.next_batch(15)
      if x_mean is not None:
        x_test = np.array([x - x_mean for x in x_test])
      x_recon = sess.run(self.y, feed_dict={self.x: x_test})

      fig, ax = plt.subplots(15, 2, figsize=(3, 10))
      for i in range(15):
        ax[i][0].imshow(np.reshape(x_test[i, :] + x_mean, (28, 28)))
        ax[i][1].imshow(np.reshape(x_recon[i, :] + x_mean, (28, 28)))

      fig.show()
      plt.draw()
      plt.waitforbuttonpress()

if __name__ == '__main__':
  import tensorflow.examples.tutorials.mnist.input_data as input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  ae = Autoencoder(28 * 28, hidden_dims=[256, 64])
  x_mean = np.mean(mnist.train.images, axis=0)
  ae.fit(mnist, x_mean)