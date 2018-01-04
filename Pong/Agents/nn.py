from pong_image import *
import gym
import tensorflow as tf
import numpy as np
import random
from Agents.parameters import *

class NeuralNetwork(object):

    def __init__(self, network_structure, input_init = False):

        #hyperparameters
        self.input_dimensions = network_structure[0]    # number of inputs/ size of after image process
        self.num_layer_neurons = network_structure      # number of neurons in each layer
        self.num_layers = len(network_structure)-1      # number of layers

        # if input nodes to NN not initialized, then assume fully connected NN
        if not input_init:
            self.layers = []
            self.input_nn = tf.placeholder(tf.float32,
                                           [None, self.input_dimensions])

        # initialize first layer with input nodes
        layer_info = self.layer_init(self.input_nn,
                                    network_structure[0],
                                    network_structure[1],
                                    type='relu',
                                    name='1')
        self.layers.append(layer_info)

        # setup hidden layers of NN
        for i in range(1, self.num_layers - 1):
            layer_info = self.layer_init(self.layers[i-1]['Qout'],
                                        network_structure[i],
                                        network_structure[i+1],
                                        type='relu',
                                        name=str(i+1))
            self.layers.append(layer_info)

        #initialize last hidden layer to output
        layer_info = self.layer_init(self.layers[-1]['Qout'],
                                    network_structure[-2],
                                    network_structure[-1],
                                    name=str(self.num_layers-1))
        self.layers.append(layer_info)

    """
    layer_init
    brief: initializes layer for neural network in TensorFlow
    input: input, num_inputs, num_outputs, learning_rate, name='-1', type=INIT_TYPE
    output: dictionary of information of layer 
    """
    def layer_init(self,input, num_inputs, num_outputs, type=INIT_TYPE, name='-1'):
        info = {}

        # initialize weight variable for layer
        W = self.weight_init(num_inputs, num_outputs, name)
        info['W'] = W

        # initialize bias variable for layer
        b = self.bias_init(num_outputs, name)
        info['b'] = b

        # initialize activation function
        if type == 'relu':
            activation = tf.nn.relu(tf.add(tf.matmul(input, W), b), name= 'relu' + name)
        elif type == 'sigmoid':
            activation = tf.nn.sigmoid(tf.add(tf.matmul(input, W), b), name= 'sigmoid' + name)
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
        xavier_var = 2 / np.sqrt(num_inputs)
        return tf.Variable(tf.random_normal([num_inputs, num_outputs],
                                            0,
                                            INIT_WEIGHT * xavier_var,
                                            dtype=tf.float32),
                           'weight' + name)

    """
    bias_init
    brief: initialize bias variable TensorFlow, start all at zero
    input: num_outputs
    output:TensorFlow variable b 
    """
    def bias_init(self,num_outputs, name):
        return tf.Variable(tf.zeros([num_outputs]), name='bias' + name)
