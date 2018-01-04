from Agents.nn import NeuralNetwork
import numpy as np
from pong_image import *
import tensorflow as tf
from Agents.parameters import *

"""
/**** NNAgent ****/
By: Edward Guevara 
References:  Sources
Description: Standard Neural Network Agent
"""
class NNAgent(NeuralNetwork):
    """Constructor for NNTAgent"""
    def __init__(self, num_layers_neurons, sess, learning_rate=1e-6, input_init = False, optimizer=INIT_OPT):

        self.sess = sess

        # initialize feedforward network
        super().__init__(num_layers_neurons, input_init=input_init)

        output_layer = self.layers[-1]

        # Loss of layer from correct output
        self.loss = tf.subtract(output_layer['y_'], output_layer['Qout'])

        # correction of loss
        self.cross_entropy = tf.reduce_mean(tf.square(self.loss))
        # self.cross_entropy = tf.reduce_mean(( (y_ * tf.log(Qout)) + ((1 - y_) * tf.log(1.0 - Qout)) ) * -1)


        # training optimization
        if optimizer == 'Adam':
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'RMSProp':
            train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=DECAY_RATE)
        else:
            train_step = None
            print("NEED train_step DEFINED")
            exit(1)
        self.model = train_step.minimize(self.cross_entropy)

        # tensorflow initialize variables
        self.init_op = tf.global_variables_initializer()

    """
    train_nn
    brief: calls model to train the neural network
    input: input, correct_output
    output:last weight (for debugging) 
    """

    def train_nn(self, input, correct_output):
        # _, W = self.sess.run([self.info[-1]['model'], [self.info[i]['W'] for i in range(0, len(self.info))]],
        #                  feed_dict={self.x: input,
        #                             self.info[-1]['y_']: correct_output})
        _ = self.sess.run(self.model,
                          feed_dict={self.input_nn: input,
                                     self.layers[-1]['y_']: correct_output})
        return

    """
    test_nn
    brief: tests the neural network
    input: input
    output:answer, accuracy 
    """

    def test_nn(self, input):
        return self.sess.run(self.layers[-1]['Qout'], feed_dict={self.input_nn: input})

    """
    test_eval
    brief: evaluates values for NNTest
    input: 
    output:none 
    """
    def test_eval(self):
        test_acc = False
        accuracy, old_acc, accOutput = 0, 0, 0
        count = 0
        num_episodes = 0
        input = [[0,0], [0,1], [1,0], [1,1]]
        output = [[0], [1], [1], [0]]
        self.sess.run(self.init_op)
        while not test_acc:
            if count == 50:
                self.sess.run(self.init_op)
                count = 0
            for i in range(0, 100):
                W = self.train_nn(input, output)
            print(num_episodes)
            # print(W)
            num_episodes+=1
            answer = self.test_nn(input)
            print(answer)
            for j, correct in enumerate(output):
                accuracy += abs(answer[j] - correct[0])
            old_acc = accOutput
            accOutput = float(1-accuracy/len(answer))
            print(accOutput)
            if accOutput > 0.9:
                test_acc = True
            accuracy = 0
            if abs(accOutput-old_acc) <0.001:
                count += 1
            else:
                count -= 0

        return accOutput, num_episodes