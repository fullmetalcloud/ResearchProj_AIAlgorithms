
import gym
import numpy as np
import tensorflow as tf

from neuralnet import *

class ConvolutionNN(NeuralNetwork):
    def __init__(self, num_hidden_layer_neurons, num_output_neurons, height, width, gamma=0.99,
                 decay_rate=0.99, learning_rate=1e-4, batch_size=10,  render_mod=100):

        self.x = tf.placeholder(tf.float32, [None, height*width])
        self.x_shaped = tf.reshape(self.x, [-1, height, width, 1])
        self.layer1 = self.create_new_conv_layer(self.x_shaped, 1, 32, [5, 5], [8, 8], name='layer1')
        self.layer2 = self.create_new_conv_layer(self.layer1, 32, 64, [5, 5], [8, 8], name='layer2')
        self.flattened = tf.reshape(self.layer2, [-1, 64])
        self.W1, self.y_1, self.Qout1, self.model1 = self.initlayer(self.flattened,
                                                                    64,
                                                                    num_hidden_layer_neurons,
                                                                    learning_rate, type='relu')

        super().__init__(height*width, num_hidden_layer_neurons, num_output_neurons, gamma=gamma,  render_mod=render_mod,
                         decay_rate=decay_rate, learning_rate=learning_rate, batch_size=batch_size, input_init=True)


    """Model functions for CNN (TensorFlow)"""
    # Input Layer
    def create_new_conv_layer(self,input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                           num_filters]

        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                              name=name + '_W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 8, 8, 1], padding='SAME')

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.nn.sigmoid(out_layer)

        # now perform max pooling
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 8, 8, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                                   padding='SAME')

        return out_layer
