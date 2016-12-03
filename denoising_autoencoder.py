from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


class DenoisingAutoencoder:
  """Simple Denoising Autoencoder"""

  def __init__(self, input_size, hidden_dims=[512, 256, 64], learning_rate=0.001,
               loss_type='squared', activation_fn=tf.nn.tanh, model_dir='./'):
    """Creates a new Autoencoder
    Args:
      input_size: int. Size of the input
      hidden_dims: array. Optional output dimensions of hidden layers
      learning_rate: float. Optional learning rate for optimizer
      batch_size: int. Optional batch size
      n_epochs: int. Optional number of epochs
      loss_type: string. Optional squared or cross_entropy loss
      activation_fn: function. Optional activation functions to use for hidden layers
    """
    self.input_size = input_size
    self.hidden_dims = hidden_dims
    self.learning_rate = learning_rate
    self.loss_type = loss_type
    self.activation_fn = activation_fn
    self.model_dir = model_dir

    self._build()

  def _corrupt(self, x):
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2,
                                               dtype=tf.int32), tf.float32))

  def _create_loss_layer(self, x, y):
    """Create Loss Layer
    Args:
      x: Tensor. Original data
      y: Tensor. Reconstructed data
    """
    with tf.name_scope('loss'):
      if self.loss_type == 'squared':
        recon_loss = tf.reduce_sum(tf.square(y - x))
      elif self.loss_type == 'cross_entropy':
        recon_loss = tf.reduce_mean(-tf.reduce_sum(
                        x * tf.log(y + 1e-10) + (1. - x) * tf.log(1. - y + 1e-10), 1))
      else:
        raise NotImplementedError()
    return recon_loss

  def _create_encoder_network(self, x):
    """Create Encoder Network
    Args:
      x: Tensor. Input to Encoder network
    Returns:
      Weights and Latent (Output) tensor
    """
    with tf.name_scope('encoder'):
      curr_in = x
      weights = []
      for layer_i, dim_output in enumerate(self.hidden_dims):
        dim_input = int(curr_in.get_shape()[1])
        with tf.name_scope('hidden_%d' % layer_i):
          W = tf.Variable(
                tf.truncated_normal([dim_input, dim_output],
                                    stddev=1.0 / math.sqrt(float(dim_input))),
                                    name='weights')
          b = tf.Variable(tf.zeros([dim_output]), name='biases')
          weights.append(W)
          h_layer = self.activation_fn(tf.matmul(curr_in, W) + b)
          curr_in = h_layer
    return weights, curr_in

  def _create_decoder_network(self, z, weights):
    """Create Decoder Network
    Args:
      z: Tensor. Latent input to the decoder network
      weights: Tensor. Resuable weights in reverse from the encoder network
    """
    with tf.name_scope('decoder'):
      curr_in = z
      dims = self.hidden_dims[:-1][::-1]
      for layer_i, dim_output in enumerate(dims):
        with tf.name_scope('hidden_%d' % layer_i):
          W = tf.transpose(weights[layer_i])
          b = tf.Variable(tf.zeros([dim_output]), name='biases')
          curr_in = self.activation_fn(tf.matmul(curr_in, W) + b)

      with tf.name_scope('recon_layer'):
        W = tf.transpose(weights[-1])
        b = tf.Variable(tf.zeros([self.input_size]), name='biases')
        recon = tf.nn.sigmoid(tf.matmul(curr_in, W) + b)
    return recon

  def _build(self):
    """Builds the Autoencoder network"""
    self.x = tf.placeholder(tf.float32, [None, self.input_size], name='x')
    self.corrupt_prob = tf.placeholder(tf.float32, [1], name='corrupt_prob')
    self.x_corrupt = self._corrupt(self.x) * self.corrupt_prob + self.x * (1 - self.corrupt_prob)

    weights, self.z = self._create_encoder_network(self.x_corrupt)
    self.y = self._create_decoder_network(self.z, weights[::-1])
    self.recon_loss = self._create_loss_layer(self.x, self.y)
    self.minimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.recon_loss)

  def fit(self, datasets, x_mean=None, n_epochs=10, batch_size=50, checkpoint_epoch=5, corrupt_prob=1.8):
    """Fit dataset
    Args:
      x_mean: np.ndarray. Optional mean of the input
      n_epochs: int. Optional number of epochs
      batch_size: int. Optional batch size
    """
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())

      for epoch in trange(n_epochs, desc='Epoch'):
        for batch_i in trange(datasets.train.num_examples // batch_size, desc='Batch'):
          x_batch, _ = datasets.train.next_batch(batch_size)
          if x_mean is not None:
            x_batch = np.array([x - x_mean for x in x_batch])
          _, loss = sess.run([self.minimize, self.recon_loss], feed_dict={self.x: x_batch,
                                                                          self.corrupt_prob: [corrupt_prob]})

        tqdm.write('\nEpoch %d: Loss = %.2f' % (epoch, loss))

        if (epoch + 1) % checkpoint_epoch == 0 or (epoch + 1) == n_epochs:
          x_test, _ = datasets.test.next_batch(15)
          if x_mean is not None:
            x_test_norm = np.array([x - x_mean for x in x_test])
          else:
            x_test_norm = x_test
          x_recon_norm = sess.run(self.y, feed_dict={self.x: x_test_norm,
                                                     self.corrupt_prob: [0.0]})

          if x_mean is not None:
            x_recon = np.array([x + x_mean for x in x_recon_norm])
          else:
            x_recon = x_recon_norm

          fig, ax = plt.subplots(15, 2, figsize=(3, 10))
          for i in range(15):
            ax[i][0].imshow(np.reshape(x_test[i, :], (28, 28)))
            ax[i][1].imshow(np.reshape(x_recon[i, :], (28, 28)))
            ax[i][0].axis('off')
            ax[i][1].axis('off')
          fig.savefig(os.path.join(self.model_dir, 'reconstruction_%08d.png' % epoch))

if __name__ == '__main__':
  import tensorflow.examples.tutorials.mnist.input_data as input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # Using cross entropy, works with softplus actication
  if not os.path.exists('output/dae_xentropy'):
    os.makedirs('output/dae_xentropy')
  ae = DenoisingAutoencoder(28 * 28, hidden_dims=[256, 64], loss_type='cross_entropy', activation_fn=tf.nn.softplus,
                            model_dir='./output/dae_xentropy')
  ae.fit(mnist, n_epochs=50, batch_size=100)

  # Using squared loss with tanh activation
  if not os.path.exists('output/dae_squared'):
    os.makedirs('output/dae_squared')
  ae = DenoisingAutoencoder(28 * 28, hidden_dims=[256, 64], loss_type='squared', activation_fn=tf.nn.tanh,
                   model_dir='./output/dae_squared')
  ae.fit(mnist, n_epochs=50, batch_size=50)