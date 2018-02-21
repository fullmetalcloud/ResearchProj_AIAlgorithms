
import tensorflow as tf
import numpy as np
from parameters import *

class Neural_Network(object):

    def __init__(self, network_structure, input_init=None, dropout=None):

        #hyperparameters
        self.input_dimensions = network_structure[0]    # number of inputs/ size of after image process
        self.network_structure = network_structure      # array with number of neurons in each layer
        self.num_layers = len(network_structure)-1      # number of layers
        self.layers = []

        if dropout is None:
            self.dropout = tf.placeholder(tf.float32)
            self.dropout_percent = DROPOUT_PERCENT
        else:
            self.dropout = dropout

        if input_init is None:
            self.input_nn = tf.placeholder(tf.float32,
                                           [None, self.input_dimensions])
        else:
            self.input_nn = input_init
        # initialize first layer with input nodes
        layer_info = self.layer_init(self.input_nn,
                                    network_structure[0],
                                    network_structure[1],
                                    type='relu',
                                    name='1',
                                     dropout=self.dropout)
        self.layers.append(layer_info)

        # setup hidden layers of NN
        for i in range(1, self.num_layers - 1):
            layer_info = self.layer_init(self.layers[i-1]['Qout'],
                                        network_structure[i],
                                        network_structure[i+1],
                                        type='relu',
                                        name=str(i+1),
                                         dropout=self.dropout)
            self.layers.append(layer_info)

        #initialize last hidden layer to output
        layer_info = self.layer_init(self.layers[-1]['Qout'],
                                    network_structure[-2],
                                    network_structure[-1],
                                    name=str(self.num_layers-1),
                                     dropout=self.dropout)
        self.layers.append(layer_info)

    """
    layer_init
    brief: initializes layer for neural network in TensorFlow
    input: input, num_inputs, num_outputs, learning_rate, name='-1', type=INIT_TYPE
    output: dictionary of information of layer 
    """
    def layer_init(self,input, num_inputs, num_outputs, type=INIT_TYPE, name='-1', dropout=None):
        info = {}

        # initialize weight variable for layer
        W = self.weight_init(num_inputs, num_outputs, name)
        info['W'] = W

        # initialize bias variable for layer
        b = self.bias_init(num_outputs, name)
        info['b'] = b

        output = tf.add(tf.matmul(input, W), b)

        output = tf.nn.dropout(output, dropout)

        # initialize activation function
        if type == 'relu':
            activation = tf.nn.relu(output, name= 'relu' + name)
        elif type == 'sigmoid':
            activation = tf.nn.sigmoid(output, name= 'sigmoid' + name)
        else:
            activation = None
            print("NEED activation function DEFINED for layer %s" % name)
            exit(1)
        info['Qout'] = activation

        # return info as dictionary
        return info

    """
    weight_init
    brief: initialize weight variable for TensorFlow
    input: num_inputs, num_outputs
    output:TensorFlow Variable W
    """
    def weight_init(self,num_inputs, num_outputs, name):
        # determine weights using Xavier Initialization
        # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        xavier_var = 1 / np.sqrt(num_inputs)
        return tf.Variable(tf.truncated_normal([num_inputs, num_outputs],
                                               stddev=INIT_WEIGHT * xavier_var),
                           'weight' + name)

    """
    bias_init
    brief: initialize bias variable TensorFlow, start all at zero
    input: num_outputs
    output:TensorFlow variable b 
    """
    def bias_init(self,num_outputs, name):
        return tf.Variable(tf.zeros([num_outputs]),
                           name='bias' + name)
