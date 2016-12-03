from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class VariationalAutoencoder:
  """Variational autoencoder"""

  def __init__(self, input_size, latent_dim, hidden_dims=[1024, 512, 256], activation_fn=tf.nn.softplus,
               learning_rate=0.001, model_dir='./'):
    self.input_size = input_size
    self.latent_dim = latent_dim
    self.hidden_dims = hidden_dims
    self.activation_fn = activation_fn
    self.learning_rate = learning_rate
    self.model_dir = model_dir

    self._build()

  def _xavier_init(self, in_dim, out_dim):
    """Xavier initilizer
    Args:
      in_dim: int. Input dimension of the variable
      out_dim: int. Output dimension of the variable
    Returns:
      Tensor for xavier initialization
    """
    return tf.random_uniform([in_dim, out_dim], -1.0 / math.sqrt(in_dim), 1.0 / math.sqrt(in_dim))

  def _montage_batch(self, images):
    """Create a montage of images with 1px border
    Args:
      images: np.ndarray. Images
    Returns:
      Montage image
    """
    count, m, n, _ = np.shape(images)
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.ones(((m + 1) * mm + 1, (n + 1) * nn + 1, 3)) * 0.5

    for i in range(mm):
      for j in range(nn):
        nth_img = i * mm + j
        if nth_img < count:
          img = images[nth_img, ...]
          M[1 + i + i * m: 1 + i + (i + 1) * m,
            1 + j + j * n: 1 + j + (j + 1) *n, :] = img

    return M

  def _create_encoder_network(self, x):
    """Create Encoder Netowrk
    Args:
      x: Tensor. Input to the encoder network
    Returns:
      Latent Tensor with its mean and log variance
    """
    with tf.name_scope('encoder'):
      curr_in = x
      with tf.name_scope('hidden'):
        for layer_i, out_dim in enumerate(self.hidden_dims):
          in_dim = int(curr_in.get_shape()[1])
          with tf.name_scope('fc_%d' % layer_i):
            W = tf.Variable(self._xavier_init(in_dim, out_dim), name='weights')
            b = tf.Variable(tf.zeros([out_dim]), name='biases')
          curr_in = self.activation_fn(tf.matmul(curr_in, W) + b)

      h = curr_in
      with tf.name_scope('latent'):
        with tf.name_scope('mu'):
          W = tf.Variable(self._xavier_init(self.hidden_dims[-1], self.latent_dim), name='weights')
          b = tf.Variable(tf.zeros([self.latent_dim]), name='biases')
          z_mean = tf.matmul(h, W) + b

        with tf.name_scope('log_sigma'):
          W = tf.Variable(self._xavier_init(self.hidden_dims[-1], self.latent_dim), name='weights')
          b = tf.Variable(tf.zeros([self.latent_dim]), name='biases')
          z_log_var = 0.5 * (tf.matmul(h, W) + b)

      epsilon = tf.random_normal(tf.pack([tf.shape(x)[0], self.latent_dim]))
      # Gaussian MLP encoder wiht L = 1
      # TODO multiple L
      z = z_mean + tf.exp(z_log_var) * epsilon
    return z, z_mean, z_log_var

  def _create_decoder_network(self, z):
    """Creates Decoder Netowrk
    Args:
      z: Tensor. Latent variable
    Returns:
      Reconstructed tensor of the data
    """
    with tf.name_scope('decoder'):
      curr_in = z
      with tf.name_scope('hidden'):
        for layer_i, out_dim in enumerate(self.hidden_dims[::-1]):
          in_dim = int(curr_in.get_shape()[1])
          with tf.name_scope('fc_%d' % layer_i):
            W = tf.Variable(self._xavier_init(in_dim, out_dim), name='weights')
            b = tf.Variable(tf.zeros([out_dim]), name='biases')
          curr_in = self.activation_fn(tf.matmul(curr_in, W) + b)

      with tf.name_scope('feature'):
        W = tf.Variable(self._xavier_init(self.hidden_dims[0], self.input_size), name='weights')
        b = tf.Variable(tf.zeros([self.input_size]), name='biases')
        y = tf.nn.sigmoid(tf.matmul(curr_in, W) + b)

    return y

  def _variational_lower_bound(self, x, y, z_mean, z_log_var):
    """Create variational lower bound tensor
    Args:
      x: Tensor. Original input data
      y: Tensor. Reconstructed data
      z_mean: Tensor. Mean of latent representation
      z_log_var: Tensor. Log Variance of latent representation
    Returns:
      Tensor of variational lower bound
    """
    with tf.name_scope('variational_lower_bound'):
      # log p(x|z)
      log_px_z = -tf.reduce_sum(
        x * tf.log(y + 1e-10) + (1. - x) * tf.log(1 - y + 1e-10), 1)

      # KL(q(z|x) || p(z))
      kl = -0.5 * tf.reduce_sum(
        1. + 2. * z_log_var - tf.square(z_mean) - tf.exp(2.0 * z_log_var), 1)

    return tf.reduce_mean(log_px_z + kl)

  def _build(self):
    """Builds the Autoencoder network"""
    self.x = tf.placeholder(tf.float32, [None, self.input_size], name='input')
    self.z, self.z_mean, self.z_log_var = self._create_encoder_network(self.x)
    self.y = self._create_decoder_network(self.z)
    self.var_lower_bound = self._variational_lower_bound(self.x, self.y, self.z_mean, self.z_log_var)

    with tf.name_scope('train'):
      self.minimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.var_lower_bound)

  def fit(self, datasets, n_epochs=50, batch_size=100, checkpoint_epoch=10, n_tests=15):
    """Fit the dataset
    Args:
      datasets: Dataset to fit
      n_epochs: int. Number of epochs
      batch_size: int. Batch size
      checkpoint_epoch: int. Epoch interval for checkpoint
      n_tests: int. Number of test samples to use for evaluation at checkpoint
    """
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      fig_manifold, ax_manifold = plt.subplots(1, 1)
      fig_recon, ax_recon = plt.subplots(n_tests, 2, figsize=(3, 10))
      fig_img_manifold, ax_img_manifold = plt.subplots(1, 1)

      for epoch in trange(n_epochs, desc='Epoch'):
        total_train_cost = 0.
        n_batches = datasets.train.num_examples // batch_size
        for batch_i in trange(n_batches, desc='Batch'):
          x_batch, _ = datasets.train.next_batch(batch_size)
          total_train_cost += sess.run([self.minimize, self.var_lower_bound], feed_dict={self.x: x_batch})[1]

        tqdm.write('Epoch %d: Variational Lower Bound = %.2f' % (epoch, total_train_cost / n_batches))

        if (epoch + 1) % checkpoint_epoch == 0 or (epoch + 1) == n_epochs:
        # if True:
          n_batches = datasets.validation.num_examples // batch_size
          total_valid_cost = 0.
          for batch_i in trange(n_batches, desc='Validation Batch'):
            x_batch, _ = datasets.validation.next_batch(batch_size)
            total_valid_cost += sess.run(self.var_lower_bound, feed_dict={self.x: x_batch})
          tqdm.write('Epoch %d: Validation variational lower bound: %.2f' % (epoch, total_valid_cost / n_batches))

          imgs = []
          for img_r in np.linspace(-3, 3, n_tests):
            for img_c in np.linspace(-3, 3, n_tests):
              z = np.array([[img_r, img_c]], dtype=np.float32)
              recon = sess.run(self.y, feed_dict={self.z: z})
              imgs.append(np.reshape(recon, (1, 28, 28, 1)))

          imgs = np.concatenate(imgs)
          ax_manifold.imshow(self._montage_batch(imgs))
          fig_manifold.savefig(os.path.join(self.model_dir, 'manifold_%08d.png' % epoch))

          xs, ys = datasets.test.images, datasets.test.labels
          zs = sess.run(self.z, feed_dict={self.x: xs})
          ax_img_manifold.clear()
          ax_img_manifold.scatter(zs[:, 0], zs[:, 1], c=np.argmax(ys, 1), alpha=0.2)
          ax_img_manifold.set_xlim([-6, 6])
          ax_img_manifold.set_ylim([-6, 6])
          ax_img_manifold.axis('off')
          fig_img_manifold.savefig(os.path.join(self.model_dir, 'image_manifold_%08d.png' % epoch))

          x_test, _ = datasets.test.next_batch(n_tests)
          x_recon = sess.run(self.y, feed_dict={self.x: x_test})
          for i in range(n_tests):
            ax_recon[i][0].imshow(np.reshape(x_test[i, :], (28, 28)))
            ax_recon[i][1].imshow(np.reshape(x_recon[i, :], (28, 28)))
            ax_recon[i][0].axis('off')
            ax_recon[i][1].axis('off')
          fig_recon.savefig(os.path.join(self.model_dir, 'recon_%08d.png' % epoch))


if __name__ == '__main__':
  import tensorflow.examples.tutorials.mnist.input_data as input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  if not os.path.exists('output/vae'):
    os.makedirs('output/vae')

  vae = VariationalAutoencoder(28 * 28, 2, model_dir='output/vae')
  vae.fit(mnist)