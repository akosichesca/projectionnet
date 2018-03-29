# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
#tf.app.flags.DEFINE_integer('BATCHSIZE', 128,
#                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_integer('TRAIN_ITER', 10000,
#                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_integer('TEST_ITER', 5000,
#                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_integer('LSH_T', 100,
#                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_integer('LSH_D', 4,
#                            """Number of images to process in a batch.""")
#FLAGS = flags.FLAGS
#print(FLAGS.LSH_T)
#FLAGS.BATCHSIZE = 100
#FLAGS.TRAIN_ITER = 10000
#FLAGS.TEST_ITER = 5000
#FLAGS.LSH_T = 100
#FLAGS.LSH_D = 4

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def random_projections(T,d,x):
  RandomPlanes = tf.truncated_normal([1,T,d,1], stddev=0.01)
  #RandomPlanes = tf.Variable(tf.random_normal([1,T,d,1], mean = 0, stddev = 1000))
  RandomPlanes2 = tf.tile(RandomPlanes,[FLAGS.BATCHSIZE,1,1,784])
  return RandomPlanes2

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  with tf.name_scope('LSH'):
    T = FLAGS.LSH_T
    d = FLAGS.LSH_D
    RandomPlanes = random_projections(T,d,x) 
    print(RandomPlanes)
    xtile = tf.tile(tf.reshape(x,[-1,1,784,1]), [1,T,1,1])
    print(xtile)
    ProjectedValues = tf.squeeze(tf.sign(tf.matmul(RandomPlanes, xtile)))
    print(ProjectedValues)
    ProjectedFlat= tf.reshape(ProjectedValues, [-1,T*d])

    W_fcP = weight_variable([T*d, 10]) #size of figure, numclasses
    b_fcP = bias_variable([10]) #numclasses
    y_P = tf.matmul(ProjectedFlat, W_fcP) + b_fcP
    P_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_P)
    P_loss = tf.reduce_sum(P_loss)

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('joint_entropy'):
    E_loss = -tf.reduce_sum(y_conv * tf.log(y_P),reduction_indices=[1])  
    E_loss = tf.reduce_sum(E_loss)

  with tf.name_scope('cross_entropy_trainer'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_sum(cross_entropy) #+ P_loss + 0.01*tf.log(E_loss)

  #W_loss = weight_variable([3,1])
  #loss = tf.reshape([cross_entropy, P_loss, E_loss], [1,3])
  #y_loss = tf.matmul(loss, W_loss)
  with tf.name_scope('loss'):
    loss = cross_entropy + P_loss + .1*E_loss

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.LEARNINGRATE).minimize(loss)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
  
  with tf.name_scope('accuracy_lsh'):
    correct_prediction2 = tf.equal(tf.argmax(y_P, 1), y_)
    correct_prediction2 = tf.cast(correct_prediction2, tf.float32)
    accuracy_P = tf.reduce_mean(correct_prediction2)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(FLAGS.TRAIN_ITER):
      batch = mnist.train.next_batch(FLAGS.BATCHSIZE)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        
        
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.9})

      if i % 100 == 0:
        #batch_test = mnist.test.next_batch(FLAGS.BATCHSIZE)
        print('test accuracy %f' % accuracy_P.eval(feed_dict={
              x:batch[0] , y_: batch[1], keep_prob: 1.0}))

    acc = 0
    for i in range(FLAGS.TEST_ITER):
        batch_test = mnist.test.next_batch(FLAGS.BATCHSIZE)
        acc_temp = accuracy_P.eval(feed_dict={
              x:batch_test[0] , y_: batch_test[1], keep_prob: 1.0})
        acc = acc + acc_temp 
        #print('test accuracy %f', acc_temp)
    print(acc/FLAGS.TEST_ITER)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--LEARNINGRATE', type=float,
                      default='1e-4',
                      help='Directory for storing input data')
  parser.add_argument('--LSH_T', type=int,
                      default=60,
                      help='Directory for storing input data')
  parser.add_argument('--LSH_D', type=int,
                      default=15,
                      help='Directory for storing input data')
  parser.add_argument('--BATCHSIZE', type=int,
                      default=128,
                      help='Directory for storing input data')
  parser.add_argument('--TRAIN_ITER', type=int,
                      default=10000,
                      help='Directory for storing input data')
  parser.add_argument('--TEST_ITER', type=int,
                      default=1000,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


