
import gym
import tensorflow as tf
import numpy as np
import random
import time
import os

from pong_image import *

def sigmoid(x):
    #sigmoid function
    return 1.0 / (1.0 + np.exp(-x))


def relu(vector):
    #rectified linear unit (reLU): x<0 = 0 else x = vector value
    vector[vector < 0] = 0
    return vector

class NeuralNetwork(object):

    def __init__(self, num_layer_neurons, gamma=0.99, decay_rate=0.99, learning_rate=1e-4,
                 batch_size=10, render_mod=100, input_init = False):
        #hyperparameters
        self.input_dimensions = num_layer_neurons[0]    # number of inputs/ size of after image process
        self.num_layer_neurons = num_layer_neurons      # number of neurons in each layer
        self.num_layers = len(num_layer_neurons)        # number of layers
        self.batch_size = batch_size                    # number of batches before learning
        self.render_mod = render_mod                    # number of episodes before rendering
        self.gamma = gamma                              # gamma for discount reward
        self.decay_rate = decay_rate                    # decay rate of learning
        self.learning_rate = learning_rate              # learning rate of weights
        self.running_reward = None                      # running reward after episodes
        self.reward_sum = 0                             # reward sum after every episode
        self.episode_num = 0                            # episode number

        # if input nodes to NN not initialized, then assume fully connected NN
        if not input_init:
            self.info = []
            self.x = tf.placeholder(tf.float32, [None, self.input_dimensions])
            info = self.initlayer(self.x,
                               num_layer_neurons[0],
                               num_layer_neurons[1],
                               learning_rate, type='relu', name='1')

            self.info.append(info)

        for i in range(1, self.num_layers - 2):
             info = self.initlayer(self.info[i-1]['Qout'+str(i)],
                                   num_layer_neurons[i],
                                   num_layer_neurons[i+1],
                                   learning_rate, type='relu',name=str(i+1))
             self.info.append(info)

        #initialize hidden layer to output
        info = self.initlayer(self.info[-1]['Qout'+str(self.num_layers-2)],
                              num_layer_neurons[-2],
                              num_layer_neurons[-1],
                              learning_rate,
                              type='sigmoid', name=str(self.num_layers-1))
        self.info.append(info)

        #tensorflow initialize variables
        self.init_op = tf.global_variables_initializer()

        self.episodes_gradient_discounted = []
    # initializes layer for NN
    def initlayer(self, input, numInputs, numOutputs, learning_rate, name, type='relu'):
        info = {}
        W = tf.Variable(tf.random_uniform([numInputs, numOutputs]) / np.sqrt(numInputs), 'weight' + name)
        info['W' + name] = W
        y_ = tf.placeholder(tf.float32, [None, numOutputs], 'output' + name)
        info['y_' + name] = y_
        if type == 'relu':
            Qout = tf.nn.relu(tf.matmul(input, W))
        elif type == 'sigmoid':
            Qout = tf.nn.sigmoid(tf.matmul(input, W))
        else:
            print("NEED Qout function DEFINED")
            exit(1)
        info['Qout' + name] = Qout
        # cross_entropy = tf.reduce_sum(tf.square(y_ - Qout))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(Qout), reduction_indices=[1])**2)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        info['model' + name] = train_step.minimize(cross_entropy)
        return info

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

    def discount_rewards(self,rewards, gamma):
        """ Actions you took 20 steps before the end result are less important to the overall result than an action 
        you took a step ago. This implements that logic by discounting the reward on previous actions based on how 
        long ago they were taken"""
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


    def discount_with_rewards(self,gradient_log_p, episode_rewards, gamma):
        """ discount the gradient with the normalized rewards """
        discounted_episode_rewards = self.discount_rewards(episode_rewards, gamma)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return gradient_log_p * discounted_episode_rewards

    def eval(self, env, sess, render=False):
        observation = env.reset()
        self.reward_sum = 0
        done = False
        episode_gradient_log_ps, episode_rewards = [], []
        w2, processed_observations, hidden_layer_values, W = [], [], [], []
        prev_processed_observations = None

        sess.run(self.init_op)
        while not done:
            if render and self.episode_num % self.render_mod == 0:
                env.render()
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations,
                                                                                          self.input_dimensions)

            up_probability = sess.run(self.info[-1]['Qout'+ str(self.num_layers-1)],
                                           feed_dict={self.x: [processed_observations]})

            action = self.choose_action(up_probability)

            # carry out the chosen action
            observation, reward, done, info = env.step(action)

            self.reward_sum += reward
            episode_rewards.append(reward)

            # see here: http://cs231n.github.io/neural-networks-2/#losses
            fake_label = 1 if action == 2 else 0
            loss_function_gradient = fake_label - up_probability
            episode_gradient_log_ps.append(loss_function_gradient)

        self.episode_num += 1

        # Combine the following values for the episode
        episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
        episode_rewards = np.vstack(episode_rewards)

        # Tweak the gradient of the log_ps based on the discounted rewards
        episode_gradient_log_ps_discounted = self.discount_with_rewards(episode_gradient_log_ps,
                                                                        episode_rewards, self.gamma)
        self.episodes_gradient_discounted.append(episode_gradient_log_ps_discounted)

        # Sum the gradient for use when we hit the batch size
        if self.episode_num % self.batch_size == 0:
            for correction in random.sample(self.episodes_gradient_discounted, int(self.batch_size/2)):
                Q, _ = sess.run([self.info[-1]['Qout'+ str(self.num_layers-1)],
                                      self.info[-1]['model' + str(self.num_layers-1)]],
                                     feed_dict={self.x: [processed_observations],
                                                self.info[-1]['y_' + str(self.num_layers-1)]: correction})

                for i in range(self.num_layers-2, 0):
                    Q, _ = sess.run([self.info[i]['Qout' + str(i)],
                                          self.info[i]['model' + str(i)]],
                                         feed_dict={self.x: [processed_observations],
                                                    self.info[i]['y_' + str(self.num_layers-i)]: Q})

            self.episodes_gradient_discounted = []
        self.running_reward = self.reward_sum if self.running_reward is None \
                              else self.running_reward * 0.99 + self.reward_sum * 0.01
        print
        'resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward)
