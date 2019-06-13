#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The TensorFlow version of neural network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
from edward.models import Normal
import numpy as np

import numpy as np
import tensorflow as tf
from getEmbeddings import getEmbeddings
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os.path

IN_DIM = 300
CLASS_NUM = 2
LEARN_RATE = 0.0001
TRAIN_STEP = 20000
tensorflow_tmp = "tmp_tensorflow"

tf.flags.DEFINE_integer("N", default=16608, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=IN_DIM, help="Number of features.")

FLAGS = tf.flags.FLAGS

def data():
    # Get the training and testing data from getEmbeddings
    if not os.path.isfile('./xtr.npy') or \
        not os.path.isfile('./xte.npy') or \
        not os.path.isfile('./ytr.npy') or \
        not os.path.isfile('./yte.npy'):
        xtr,xte,ytr,yte = getEmbeddings("datasets/train.csv")
        np.save('./xtr', xtr)
        np.save('./xte', xte)
        np.save('./ytr', ytr)
        np.save('./yte', yte)
    # Read the Doc2Vec data
    train_data = np.load('./xtr.npy')
    eval_data = np.load('./xte.npy')
    train_labels = np.load('./ytr.npy')
    eval_labels = np.load('./yte.npy')
    train_labels = train_labels.reshape((-1, 1)).astype(np.int32)
    #eval_labels = eval_labels.reshape((-1, 1)).astype(np.int32)
   
    return train_data, train_labels.flatten() 

def neural_network(X):
    h = tf.nn.relu(tf.matmul(X, W_0) + b_0)
    h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
    h = tf.nn.relu(tf.matmul(h, W_2) + b_2)
    h = tf.nn.relu(tf.matmul(h, W_3) + b_3)
    return tf.reshape(h, [-1])

ed.set_seed(42)
#
with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([FLAGS.D, 300]), scale=tf.ones([FLAGS.D, 300]),
                 name="W_0")
    W_1 = Normal(loc=tf.zeros([300, 300]), scale=tf.ones([300, 300]), name="W_1")
    W_2 = Normal(loc=tf.zeros([300, 300]), scale=tf.ones([300, 300]), name="W_2")
    W_3 = Normal(loc=tf.zeros([300, 1]), scale=tf.ones([300, 1]), name="W_3")
    b_0 = Normal(loc=tf.zeros(300), scale=tf.ones(300), name="b_0")
    b_1 = Normal(loc=tf.zeros(300), scale=tf.ones(300), name="b_1")
    b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(300), name="b_2")
    b_3 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")
    X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D], name="X")
    y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(FLAGS.N), name="y")
#
#
X_train, y_train = data()
#
# INFERENCE
with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
      loc = tf.get_variable("loc", [FLAGS.D, 300])
      scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.D, 300]))
      qW_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_1"):
      loc = tf.get_variable("loc", [300, 300])
      scale = tf.nn.softplus(tf.get_variable("scale", [300, 300]))
      qW_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_2"):
      loc = tf.get_variable("loc", [300, 300])
      scale = tf.nn.softplus(tf.get_variable("scale", [300, 300]))
      qW_2 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_3"):
      loc = tf.get_variable("loc", [300, 1])
      scale = tf.nn.softplus(tf.get_variable("scale", [300, 1]))
      qW_3 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_0"):
      loc = tf.get_variable("loc", [300])
      scale = tf.nn.softplus(tf.get_variable("scale", [300]))
      qb_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_1"):
      loc = tf.get_variable("loc", [300])
      scale = tf.nn.softplus(tf.get_variable("scale", [300]))
      qb_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_2"):
      loc = tf.get_variable("loc", [300])
      scale = tf.nn.softplus(tf.get_variable("scale", [300]))
      qb_2 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_3"):
      loc = tf.get_variable("loc", [1])
      scale = tf.nn.softplus(tf.get_variable("scale", [1]))
      qb_3 = Normal(loc=loc, scale=scale)
    #
    #
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                   W_1: qW_1, b_1: qb_1,
                   W_2: qW_2, b_2: qb_2,
                   W_3: qW_3, b_3: qb_3}, data={X: X_train, y: y_train})
inference.run(logdir='log')
