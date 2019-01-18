import tensorflow as tf
import numpy as np
import random


state_size = [200, 200, 1]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = 100             # 3 possible actions: left, right, shoot
learning_rate = 0.0002      # Alpha (aka learning rate)

# TRAINING HYPER PARAMETERS
total_episodes = 500        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyper parameters
gamma = 0.95               # Discounting rate

# MEMORY HYPER PARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

# TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


class Cnn:
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope('Cnn'):
            self.inputs = tf.placeholder(tf.float32, [None, *state_size])
            self.actions = tf.placeholder(tf.float32, [None, action_size])

            self.sample_op = tf.placeholder(tf.float32, [None])

            self.conv1 = tf.layers.conv2d(inputs=self.inputs, filters=32, kernel_size=[10, 10], strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.conv1_batch_norm = tf.layers.batch_normalization(self.conv1, training=True, epsilon=1e-5)

            self.conv1_out = tf.nn.relu(self.conv1_batch_norm)

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[5, 5], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.conv2_batch_norm = tf.layers.batch_normalization(self.conv2, training=True, epsilon=1e-5)

            self.conv2_out = tf.nn.relu(self.conv2_batch_norm)

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=128, kernel_size=[5, 5], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.conv3_batch_norm = tf.layers.batch_normalization(self.conv3, training=True, epsilon=1e-5)

            self.conv3_out = tf.nn.relu(self.conv3_batch_norm)

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten, units=1000, activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.output = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3, activation=None)

            self.pred_op = tf.reduce_sum(tf.multiply(self.output, self.actions))

            self.loss = tf.reduce_mean(tf.square(self.sample_op - self.pred_op))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
