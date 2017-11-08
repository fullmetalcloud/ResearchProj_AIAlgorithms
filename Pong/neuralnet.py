from pong_image import *
import gym
import tensorflow as tf
import numpy as np
import random

INIT_WEIGHT = 0.5

def sigmoid(x):
    #sigmoid function
    return 1.0 / (1.0 + np.exp(-x))


def relu(vector):
    #rectified linear unit (reLU): x<0 = 0 else x = vector value
    vector[vector < 0] = 0
    return vector

class NeuralNetwork(object):

    def __init__(self, num_layer_neurons, sess, learning_rate=1e-3, batch_size=10,
                 render_mod=100, input_init = False):
        #hyperparameters
        self.input_dimensions = num_layer_neurons[0]  # number of inputs/ size of after image process
        self.num_layer_neurons = num_layer_neurons      # number of neurons in each layer
        self.num_layers = len(num_layer_neurons)        # number of layers
        self.batch_size = batch_size                    # number of batches before learning
        self.render_mod = render_mod                    # number of episodes before rendering
        self.learning_rate = learning_rate              # learning rate of weights
        self.reward_sum = 0                             # reward sum after every episode
        self.episode_num = 0                            # episode number
        self.sess = sess                                # session for TensorFlow

        # if input nodes to NN not initialized, then assume fully connected NN
        if not input_init:
            self.info = []
            self.input_nn = tf.placeholder(tf.float32, [None, self.input_dimensions])
            self.x = self.input_nn
        for i in range(0, self.num_layers - 2):
            if i == 0:
                info = self.initlayer(self.input_nn,
                                      num_layer_neurons[0],
                                      num_layer_neurons[1],
                                      learning_rate, type='relu', name='1')
            else:
                info = self.initlayer(self.info[i-1]['Qout'],
                                      num_layer_neurons[i],
                                      num_layer_neurons[i+1],
                                      learning_rate, type='relu',name=str(i+1))
            self.info.append(info)

        #initialize hidden layer to output
        info = self.initlayer(self.info[-1]['Qout'],
                              num_layer_neurons[-2],
                              num_layer_neurons[-1],
                              learning_rate,
                              type='sigmoid', name=str(self.num_layers-1))
        self.info.append(info)

        #tensorflow initialize variables
        self.init_op = tf.global_variables_initializer()

        self.episodes_gradient_discounted, self.episodes_actions, self.episodes_rewards, self.fake_labels = [], [], [], []
    # initializes layer for NN
    def initlayer(self, input, numInputs, numOutputs, learning_rate, name='-1', type='relu'):
        info = {}
        W = tf.Variable(tf.random_uniform([numInputs, numOutputs], -INIT_WEIGHT, INIT_WEIGHT), 'weight' + name)
        info['W'] = W
        b = tf.Variable(tf.random_uniform([numOutputs], -INIT_WEIGHT, INIT_WEIGHT))
        y_ = tf.placeholder(tf.float32, [None, numOutputs], 'output' + name)
        info['y_'] = y_
        if type == 'relu':
            Qout = tf.nn.relu(tf.matmul(input, W)+b)
        elif type == 'sigmoid':
            Qout = tf.nn.sigmoid(tf.matmul(input, W)+b)
        else:
            print("NEED Qout function DEFINED")
            exit(1)
        info['Qout'] = Qout
        # cross_entropy = tf.reduce_sum(tf.square(y_ - Qout))
        cross_entropy = tf.reduce_mean(( (y_ * tf.log(Qout)) + ((1 - y_) * tf.log(1.0 - Qout)) ) * -1)
        info['accuracy'] = cross_entropy
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        info['model'] = train_step.minimize(cross_entropy)
        return info

    """
    train_nn
    brief: calls model to train the neural network
    input: input, correct_output
    output:last weight (for debugging) 
    """
    def train_nn(self,input, correct_output):
        _, W1 = self.sess.run([self.info[-1]['model'], self.info[-1]['W']],
                         feed_dict={self.x: input,
                                    self.info[-1]['y_']: correct_output})
        return W1

    """
    test_nn
    brief: tests the neural network
    input: input
    output:answer, accuracy 
    """
    def test_nn(self,input):
        answer = self.sess.run(self.info[-1]['Qout'], feed_dict={self.x: input})
        return answer

    # chooses action based on single output
    def choose_action(self,probability):
        random_value = np.random.uniform()
        # random_value = 0.5
        if random_value < probability:
            # signifies up in openai gym
            return 2
        else:
            # signifies down in openai gym
            return 3

    def eval(self, env, render=False):
        observation = env.reset()
        self.reward_sum = 0
        done = False
        episode_rewards, episode_observations, fake_labels = [], [], []
        prev_processed_observations = None

        self.sess.run(self.init_op)
        while not done:
            if render and self.episode_num % self.render_mod == 0:
                env.render()
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations,
                                                                                          self.input_dimensions)

            episode_observations.append(processed_observations)

            up_probability = self.test_nn([processed_observations])

            action = self.choose_action(up_probability)

            # carry out the chosen action
            observation, reward, done, info = env.step(action)

            self.reward_sum += reward

            # see here: http://cs231n.github.io/neural-networks-2/#losses
            fake_label = 1 if action == 2 else 0
            fake_labels.append(fake_label)

        self.episode_num += 1

        # Combine the following values for the episode
        self.fake_labels = np.vstack(fake_labels)
        episode_observations = np.vstack(episode_observations)

        self.episodes_rewards.append(self.reward_sum)
        self.episodes_actions.append(episode_observations)
        # Sum the gradient for use when we hit the batch size
        if self.episode_num % self.batch_size == 0:
            best_rewards = sorted(range(len(self.episodes_rewards)),
                                  key=lambda x: self.episodes_rewards[x])[-int(self.batch_size/2):]
            for i in best_rewards:
                for j, correct_action in enumerate(self.fake_labels[i]):
                    Q = [[correct_action]]
                    W = self.train_nn([self.episodes_actions[i][j]], Q)
        return

"""
/**** NNTest ****/
By: Edward Guevara 
References:  Sources
Description: Test the class Neural Network using XOR example
"""
class NNTest(NeuralNetwork):
    """Constructor for NNTest"""
    def __init__(self, train_episodes, sess):
        self.train_episodes = train_episodes
        layers = [2, 2, 1]


        super().__init__(layers, sess, learning_rate=1e-1)

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
            for i in range(0, self.train_episodes):
                W = self.train_nn(input, output)
            # print(num_episodes)
            num_episodes+=1
            answer = self.test_nn(input)
            # print(answer)
            for j, correct in enumerate(output):
                accuracy += abs(answer[j] - correct[0])
            old_acc = accOutput
            accOutput = float(1-accuracy/len(answer))
            # print(accOutput)
            if accOutput > 0.9:
                test_acc = True
            accuracy = 0
            if abs(accOutput-old_acc) <0.001:
                count += 1
            else:
                count -= 0

        return accOutput, num_episodes
