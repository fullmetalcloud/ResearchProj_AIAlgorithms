
import gym
import tensorflow as tf
import numpy as np
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

    def __init__(self, input_dimensions, num_hidden_layer_neurons, num_output_neurons, gamma=0.99,
                 decay_rate=0.99, learning_rate=1e-3, batch_size=10, render_mod=100, input_init = False):
        self.batch_size = batch_size
        self.render_mod = render_mod
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.input_dimensions = input_dimensions
        self.learning_rate = learning_rate
        self.running_reward = None
        self.reward_sum = 0
        self.episode_num = 0

        if not input_init:
            self.x = tf.placeholder(tf.float32, [None, input_dimensions])
            self.W1, self.y_1, self.Qout1, self.model1 = self.initlayer(self.x,
                                                                        input_dimensions,
                                                                        num_hidden_layer_neurons,
                                                                        learning_rate, type='relu')
        self.W2, self.y_2,self.Qout2, self.model2 = self.initlayer(self.Qout1,
                                                                   num_hidden_layer_neurons,
                                                                   num_output_neurons,
                                                                   learning_rate, type='sigmoid')
        self.init_op = tf.initialize_all_variables()

        self.episode_gradient_log_ps = []
        self.episode_rewards = []

    def initlayer(self, input, numInputs, numOutputs, learning_rate, type='relu'):
        W = tf.Variable(tf.random_uniform([numInputs, numOutputs], -0.05, 0.05))
        y_ = tf.placeholder(tf.float32, [None, numOutputs])
        if type == 'relu':
            Qout = tf.nn.relu(tf.matmul(input, W))
        elif type == 'sigmoid':
            Qout = tf.nn.sigmoid(tf.matmul(input, W))
        else:
            print("NEED Qout function DEFINED")
            exit(1)
        cross_entropy = tf.reduce_sum(tf.square(y_ - Qout))
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(Qout), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        updateModel = train_step.minimize(cross_entropy)
        return W, y_, Qout, updateModel

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
        """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
        This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
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
        w2, processed_observations, hidden_layer_values, W_1, W_2 = [], [], [], [], []
        prev_processed_observations = None

        sess.run(self.init_op)
        while not done:
            if render and self.episode_num % self.render_mod == 0:
                env.render()
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations,
                                                                                          self.input_dimensions)

            hidden_layer_values, up_probability, w2 = sess.run([self.Qout1, self.Qout2, self.W2],
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

        # Sum the gradient for use when we hit the batch size
        if self.episode_num % self.batch_size == 0:
            _, W_1, W_2 = sess.run([self.model2, self.W1, self.W2],
                              feed_dict={self.x: [processed_observations],
                                         self.y_2: episode_gradient_log_ps_discounted})

        self.running_reward = self.reward_sum if self.running_reward is None \
                              else self.running_reward * 0.99 + self.reward_sum * 0.01
        print
        'resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward)
