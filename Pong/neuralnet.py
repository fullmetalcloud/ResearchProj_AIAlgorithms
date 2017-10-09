
import gym
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

    def __init__(self,  num_hidden_layer_neurons, input_dimensions, gamma=0.99,
                 decay_rate=0.99, learning_rate=1e-4, batch_size=10, render_mod=100):
        self.batch_size = batch_size
        self.render_mod = render_mod
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.input_dimensions = input_dimensions
        self.learning_rate = learning_rate
        self.running_reward = None
        self.reward_sum = 0
        self.episode_num = 0

        self.weights = {
            '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
            '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
        }

        self.expectation_g_squared = {}
        self.g_dict = {}
        for layer_name in self.weights.keys():
            self.expectation_g_squared[layer_name] = np.zeros_like(self.weights[layer_name])
            self.g_dict[layer_name] = np.zeros_like(self.weights[layer_name])

        self.episode_hidden_layer_values = []
        self.episode_observations = []
        self.episode_gradient_log_ps = []
        self.episode_rewards = []

    def apply_neural_nets(self,observation_matrix):
        """ Based on the observation_matrix and weights, compute the new 
        hidden layer values and the new output layer values"""
        hidden_layer_values = np.dot(self.weights['1'], observation_matrix)
        hidden_layer_values = relu(hidden_layer_values)
        output_layer_values = np.dot(hidden_layer_values, self.weights['2'])
        output_layer_values = sigmoid(output_layer_values)
        return hidden_layer_values, output_layer_values


    def choose_action(self,probability):
        random_value = np.random.uniform()
        if random_value < probability:
            # signifies up in openai gym
            return 2
        else:
            # signifies down in openai gym
            return 3


    def compute_gradient(self,gradient_log_p, hidden_layer_values, observation_values, weights):
        """ See here: http://neuralnetworksanddeeplearning.com/chap2.html"""
        delta_L = gradient_log_p
        dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
        delta_l2 = np.outer(delta_L, weights['2'])
        delta_l2 = relu(delta_l2)
        dC_dw1 = np.dot(delta_l2.T, observation_values)
        return {
            '1': dC_dw1,
            '2': dC_dw2
        }


    def update_weights(self,weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
        """ See here: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
        epsilon = 1e-5
        for layer_name in weights.keys():
            g = g_dict[layer_name]
            expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g ** 2
            weights[layer_name] += (learning_rate * g) / (np.sqrt(expectation_g_squared[layer_name] + epsilon))
            g_dict[layer_name] = np.zeros_like(weights[layer_name])  # reset batch gradient buffer


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

    def eval(self, env, render=False):
        observation = env.reset()
        self.reward_sum = 0
        done = False
        episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
        prev_processed_observations = None
        while not done:
            if render and self.episode_num % self.render_mod == 0:
                env.render()
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations,
                                                                                          self.input_dimensions)
            hidden_layer_values, up_probability = self.apply_neural_nets(processed_observations)

            episode_observations.append(processed_observations)
            episode_hidden_layer_values.append(hidden_layer_values)

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
        episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
        episode_observations = np.vstack(episode_observations)
        episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
        episode_rewards = np.vstack(episode_rewards)

        # Tweak the gradient of the log_ps based on the discounted rewards
        episode_gradient_log_ps_discounted = self.discount_with_rewards(episode_gradient_log_ps,
                                                                        episode_rewards, self.gamma)

        gradient = self.compute_gradient(
            episode_gradient_log_ps_discounted,
            episode_hidden_layer_values,
            episode_observations,
            self.weights
        )

        # Sum the gradient for use when we hit the batch size
        for layer_name in gradient:
            self.g_dict[layer_name] += gradient[layer_name]

        if self.episode_num % self.batch_size == 0:
            self.update_weights(self.weights, self.expectation_g_squared,
                                self.g_dict, self.decay_rate, self.learning_rate)

        self.running_reward = self.reward_sum if self.running_reward is None \
                              else self.running_reward * 0.99 + self.reward_sum * 0.01
        print
        'resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward)