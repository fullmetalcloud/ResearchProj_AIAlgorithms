
from parameters import *
import numpy as np
import tensorflow as tf
from nn import Neural_Network

"""
/**** Convolutional_NN ****/
By: Edward Guevara 
References:  Sources
Description: initializes convolutional neural network
"""
class Convolutional_NN(object):
    """Constructor for Convolutional_NN"""
    def __init__(self, height, width, nn_structure):
        self.dropout = tf.placeholder(tf.float32)
        self.dropout_percent = DROPOUT_PERCENT
        self.input_nn = tf.placeholder(tf.float32, [None, height * width])
        self.x_shaped = tf.reshape(self.input_nn, [-1, height, width, 1])
        self.layer1 = create_new_conv_layer(self.x_shaped, 1, 8, [8, 8], [4, 4], [4, 4], [2, 2],
                                            name='layer1', dropout=self.dropout)
        self.layer2 = create_new_conv_layer(self.layer1, 8, 16, [4, 4], [2, 2], [2, 2], [2, 2],
                                            name='layer2', dropout=self.dropout)
        self.layer3 = create_new_conv_layer(self.layer2, 16, 32, [2, 2], [2, 2], [1, 1], [1, 1],
                                            name='layer2', dropout=self.dropout)
        self.x = tf.reshape(self.layer3, [tf.shape(self.layer3)[0], -1])
        self.neural_net = Neural_Network(nn_structure, self.x, dropout=self.dropout)

"""Model functions for CNN (TensorFlow)"""
# Input Layer
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, filter_stride_size,
                          pool_shape, pool_stride_size, name, dropout=1):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                       num_filters]

    # initialise weights and bias for the filter
    xavier_var = 1/np.sqrt(filter_shape[0]*filter_shape[1]*num_input_channels*num_filters)
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape,
                                              stddev= INIT_WEIGHT * xavier_var),
                          name=name + '_W')
    bias = tf.Variable(tf.zeros([num_filters]),
                       name=name + '_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data,
                             weights,
                             [1, filter_stride_size[0], filter_stride_size[1], 1],
                             padding='SAME')
    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # Dropout percent from answer
    out_layer = tf.nn.dropout(out_layer, dropout)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, pool_stride_size[0], pool_stride_size[1], 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer
