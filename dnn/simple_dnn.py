from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path

import tensorflow as tf
from tqdm import tqdm, trange

#
class SimpleDNN:
  def __init__(self, n_classes, input_size, hidden_dims=[128, 32], n_epochs=50,
               batch_size=100, learning_rate=0.01, model_dir='./', checkpoint_epoch=10):
    self.n_classes = n_classes
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.input_size = input_size
    self.learning_rate = learning_rate
    self.hidden_dims = hidden_dims
    self.checkpoint_epoch = checkpoint_epoch
    self.model_dir = model_dir
    self.layers = []

    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)

    self._build()

  def _create_hidden_layer(self, input, name, out_size, activation_fn=None):
    """Create an affine hidden layer with activation

    Args:
      input: Tensor. Input to this hidden layer
      name: string. Name of the hidden layer used to scope its variables
      out_size: int. Dimension of the output of the layer
      activation_fn: function. Optional activation function

    Returns:
      The Tensor of the output
    """

    with tf.name_scope(name):
      in_size = input.get_shape().as_list()[1]
      W = tf.Variable(
        tf.truncated_normal((in_size, out_size),
                            stddev=1.0 / math.sqrt(float(in_size))),
                            name='Weights')
      biases = tf.Variable(tf.zeros([out_size]), name='biases')
      h_layer = tf.matmul(input, W) + biases

      if activation_fn:
        h_layer = activation_fn(h_layer)

      return h_layer

  def _create_cross_entropy_layer(self, logits):
    """Create cross entopy loss layer
    Args:
      logits: Tensor. Logit input to the layer

    Returns:
      The loss tensor
    """
    with tf.name_scope('cross_entropy'):
      labels = tf.to_int64(self.labels)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits, labels, name='cross_entropy')
      with tf.name_scope('loss'):
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

    return loss

  def _build(self):
    """Builds the entire DNN"""
    self.input = tf.placeholder(tf.float32, shape=(None, self.input_size))
    self.labels = tf.placeholder(tf.int32, shape=(None))

    curr_in = self.input
    for layer_i, hdims in enumerate(self.hidden_dims):

      if layer_i + 1 < len(self.hidden_dims):
        activation = None
      else:
        activation = tf.nn.relu

      curr_in = self._create_hidden_layer(curr_in,
                  name='hidden_%d' % layer_i,
                  out_size=hdims,
                  activation_fn=activation)

      self.layers.append(curr_in)

    logits = curr_in
    self.loss = self._create_cross_entropy_layer(logits)

    with tf.name_scope('train'):
      self.minimize = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    with tf.name_scope('accuracy'):
      correct = tf.nn.in_top_k(logits, self.labels, 1)
      self.accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))

  def fit(self, datasets):
    """Fit the DNN with the given dataset

    Args:
      datasets Dataset to fit
    """
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(init)

      for epoch in trange(self.n_epochs, desc='Epoch'):
        for batch_i in trange(datasets.train.num_examples // self.batch_size, desc='Batch'):
          features, labels = datasets.train.next_batch(self.batch_size)
          _, loss = sess.run([self.minimize, self.loss], feed_dict={
            self.input: features,
            self.labels: labels
          })

        tqdm.write('\nEpoch %d: Loss = %.2f' % (epoch, loss))

        if (epoch + 1) % self.checkpoint_epoch == 0 or (epoch + 1) == self.n_epochs:
          ck_file = os.path.join(self.model_dir, 'checkpoint')
          saver.save(sess, ck_file, global_step=epoch)

          tqdm.write('Evaluation:\n===============')
          precision, total_correct, n_datapoints = self.evaluate(sess, datasets.train)
          tqdm.write('\tTrain:\t\t Precision = %.2f (%d/%d)' % (precision, total_correct, n_datapoints))

          precision, total_correct, n_datapoints = self.evaluate(sess, datasets.validation)
          tqdm.write('\tValidation:\t Precision = %.2f (%d/%d)' % (precision, total_correct, n_datapoints))

          precision, total_correct, n_datapoints = self.evaluate(sess, datasets.test)
          tqdm.write('\tTest:\t\t Precision = %.2f (%d/%d)' % (precision, total_correct, n_datapoints))

  def evaluate(self, sess, datasets):
    """Evaluates the learned model on the dataset

    Args:
      datasets: Dataset to evaluate

    Returns:
      precision, total corrent prediction and the number of data points
      evaluated
    """
    total_correct = 0
    n_batches = datasets.num_examples // self.batch_size

    for batch_i in trange(n_batches):
      features, labels = datasets.next_batch(self.batch_size)
      feed_dict = {self.input: features, self.labels: labels}
      accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
      total_correct += accuracy

    n_datapoints = n_batches * self.batch_size
    precision = total_correct / n_datapoints
    return precision, total_correct, n_datapoints

#
if __name__ == '__main__':
  from tensorflow.examples.tutorials.mnist import input_data
  datasets = input_data.read_data_sets('MNIST_data')


  simple_dnn = SimpleDNN(10, 28 * 28, model_dir='simple_dnn_mnist')
  simple_dnn.fit(datasets)