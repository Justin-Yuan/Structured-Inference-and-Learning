import numpy as np
import tensorflow as tf
# %matplotlib inline
# import matplotlib.pyplot as plt

##
# Task:
# Use a binary sequence X to predict a binary sequence Y
# Input sequence X: for step t, X_t has 50% chance of being 1, and 50% chance of being 0. E.g., [0, 1, 1, 0, 0, ...]
# Output sequence Y: Similar to X, X_t has 50% chance of being 1, and 50% chance of being 0.
#   The chance of Y_t being 1 is increased by 50% (i.e., to 100%), if X_(t-3) is 1, and decreased by 25% (i.e., to 25%) if X_(t-8) is 1.
#   If X_(t-3) and X_(t-8) are both 1, then the chance of Y_t being 1 is 50% + 50% - 25% = 75%

##
# Model architecture
# At time step t (t=0,1,2,...n), the model accepts a one-hot binary vector X_t, and a previous state vector S_(t-1), as inputs and produces a state vector S_t, and a predicted probability distribution vector, P_t, for the one-hot binary vector Y_t
# Formally
# S_t = tanh(W.dot(X_t @ S_(t-1)) + b_s)
# P_t = Softmax(U.dot(S_t) + b_p)
# where @ represents concatenation, the dim of X is 2, the dim of S is d
# the dim of W is d*(2+d), and the dim of b_s is 2
# the dim of U is 2*d, and the dim of b_p is 2


"""
setup_graph
"""
def setup_graph(graph, config):
    with graph.as_default():

        """
        Placeholders
        """
        x = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name='input_placeholder')
        y = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name='labels_placeholder')
        default_init_state = tf.zeros([config.batch_size, config.state_size])
        init_state = tf.placeholder_with_default(default_init_state, [config.batch_size, config.state_size], name='state_placeholder')


        """
        rnn_inputs and y_as_list
        """
        # Turn the x placeholder into a list of one-hot tensors
        # 1. tf.split creates a list of config.num_steps tensors, each with shape [batch_size X 1 X 2]
        # 2. tf.squeeze gets rid of the middle dimension from each
        # 3. rnn_inputs is a list of config.num_steps tensors with shape [batch_size, 2]
        x_one_hot = tf.one_hot(x, depth=config.num_classes)
        # rnn_inputs =[tf.squeeze(i, axis=[1]) for i in tf.split(value=x_one_hot, num_or_size_splits=config.num_steps, axis=1)]
        rnn_inputs = tf.unstack(x_one_hot, num=config.num_steps, axis=1)

        # Turn the y placeholder into a list of one-hot tensors
        y_one_hot = tf.one_hot(y, config.num_classes)
        # y_as_list = [tf.squeeze(i, axis=[1]) for i in tf.split(value=y_one_hot, num_or_size_splits=config.num_steps, axis=1)]
        # why not following
        y_as_list = tf.unstack(y, num=config.num_steps, axis=1)


        """
        Define rnn_cell
        """
        with tf.variable_scope('rnn_cell'):
            W = tf.get_variable('W', [config.num_classes + config.state_size, config.state_size])
            b = tf.get_variable('b', [config.state_size], initializer=tf.constant_initializer(0.0))

        def rnn_cell(rnn_input, state):
            with tf.variable_scope('rnn_cell', reuse=True):
                W = tf.get_variable('W', [config.num_classes + config.state_size, config.state_size])
                b = tf.get_variable('b', [config.state_size], initializer=tf.constant_initializer(0.0))
            return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
        # NOTE that the shape of rnn_input is (200,2), which is an element of rnn_inputs
        # and the shape of state is (200, 4)
        # tf.concat along the 1 axis results in a tensor of shape (200, 6)


        """
        Add rnn_cells to graph
        """
        state = init_state
        rnn_outputs = []
        for rnn_input in rnn_inputs:
            state = rnn_cell(rnn_input, state)
            rnn_outputs.append(state)
        final_state = rnn_outputs[-1]


        """
        Prediction, loss, and training step
        """
        # Logits and predictions
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [config.state_size, config.num_classes])
            b = tf.get_variable('b', [config.num_classes], initializer=tf.constant_initializer(0.0))
        logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
        predictions = [tf.nn.softmax(logit) for logit in logits]
        # Losses and train_step
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for logit, label in zip(logits, y_as_list)]
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(total_loss)

        return losses, total_loss, final_state, train_step, x, y, init_state



# """
# RNN inputs
# """
# turn the x placeholder into a list of one-hot tensors:
# run_inputs is a list of num_steps tensors with shape [batch_size, num_classes]

# For tf.unstack, see https://www.tensorflow.org/api_docs/python/tf/unstack
# Shape of x_one_hot: (200, 5, 2)
# rnn_inputs:
# [<tf.Tensor 'unstack:0' shape=(200, 2) dtype=float32>,
 # <tf.Tensor 'unstack:1' shape=(200, 2) dtype=float32>,
 # <tf.Tensor 'unstack:2' shape=(200, 2) dtype=float32>,
 # <tf.Tensor 'unstack:3' shape=(200, 2) dtype=float32>,
 # <tf.Tensor 'unstack:4' shape=(200, 2) dtype=float32>]


# """
# Define rnn_cell
# This is very similar to the __call__ method on Tensorflow's BasicRNNCell. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L95
# """
#
#
# """
# Add rnn_cell to graph
# """
# Turn the y placeholder into a list of labels
# y_as_list = tf.unstack(y, num=num_steps, axis=1)
# [<tf.Tensor 'unstack_1:0' shape=(200,) dtype=int32>,
 # <tf.Tensor 'unstack_1:1' shape=(200,) dtype=int32>,
 # <tf.Tensor 'unstack_1:2' shape=(200,) dtype=int32>,
 # <tf.Tensor 'unstack_1:3' shape=(200,) dtype=int32>,
 # <tf.Tensor 'unstack_1:4' shape=(200,) dtype=int32>]
