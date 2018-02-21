
import os
import numpy as np
from policy_gradient import *
from cnn import *
from pong_image import *
from parameters import *

"""
choose_action
brief: determine action to be taken
input: action
output: value
"""

def choose_action(action):
    random_value = np.random.uniform()
    if random_value < action:
        # signifies up in openai gym
        return 2
    else:
        # signifies down in openai gym
        return 3

"""
/**** Pong_PoliGrad_Agent ****/
By: Edward Guevara 
References:  Sources
Description: Agent that learns Pong using Policy Gradient optimization
"""
class Pong_PoliGrad_Agent(object):
    """Constructor for Pong_PoliGrad_Agent"""
    def __init__(self, num_layer_neurons, sess, env, render=False, learning_rate=LEARNING_RATE_DEFAULT,
                 batch_size=BATCH_SIZE_DEFAULT, render_mod=RENDER_MOD_DEFAULT, optimizer=INIT_OPT, directory=None):
        self.reward_sum = 0             # reward of an episode
        self.running_reward = None      # running reward of 100 games
        self.env = env                  # environment for OpenAI
        self.episode_num = 0            # number of episodes
        self.render = render            # if game should be rendered/shown
        self.render_mod = render_mod    # number of episodes before rendering

        # setup policy gradient neural network
        self.poli_grad_nn = Policy_Gradient_NN(num_layer_neurons,
                                               sess,
                                               learning_rate,
                                               batch_size,
                                               optimizer,
                                               directory=directory)

    """
    pong_eval
    brief: runs and learns pong game
    input: 
    output: 
    """

    def pong_eval(self):

        # reset environment and initialize vars and function calls
        observation = self.env.reset()
        self.reward_sum = 0
        done = False
        episode_rewards, episode_observations, fake_labels = [], [], []
        prev_processed_observations = None
        test_nn = self.poli_grad_nn.test_nn
        train_eval = self.poli_grad_nn.train_nn

        # check if game is rendered/shown
        check = self.render and self.episode_num % self.render_mod == 0

        # play Pong
        while not done:
            if check:
                self.env.render()

            # preprocess observation before applying
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations)
            # save input
            episode_observations.append(processed_observations)

            # determine action
            up_probability = test_nn([processed_observations])
            # action = choose_action(up_probability[0], self.chance_prob)
            action = choose_action(up_probability[0])

            # carry out the chosen action
            observation, reward, done, info = self.env.step(action)

            # save reward, decrease reward if AI loses, increase reward if AI wins
            self.reward_sum += reward
            if reward < 0:
                reward *= (1 - REWARD_CORRECTION)
            else:
                reward *= (1 + REWARD_CORRECTION)
            episode_rewards.append(reward)

            # change action to 1 == 2/up and 0 == 3/down
            # see here: http://cs231n.github.io/neural-networks-2/#losses
            fake_label = 1 if action == 2 else 0
            fake_labels.append(fake_label)

            # Episode finished/ begin updates
            if reward != 0:
                # if done:
                train_eval(np.array(episode_rewards), episode_observations, fake_labels)
                episode_rewards, episode_observations, fake_labels = [], [], []

        # update running reward
        self.running_reward = self.reward_sum if self.running_reward is None \
            else self.running_reward * 0.99 + self.reward_sum * 0.01

        # if AI wins, save network
        if self.reward_sum > 0:
            print("\n\n<<<<<<<<< I WON >>>>>>>>>\n\n")
            self.poli_grad_nn.saver.save(self.poli_grad_nn.sess, os.getcwd() + "/tmp/NN/")

        # update episode count
        self.episode_num += 1

        return

"""
/**** Pong_RNN_PoliGrad_Agent ****/
By: Edward Guevara 
References:  Sources
Description: Recurrent NN Agent that learns Pong using Policy Gradient optimization 
"""
class Pong_RNN_PoliGrad_Agent(object):
    """Constructor for Pong_PoliGrad_Agent"""
    def __init__(self, num_layer_neurons, sess, env, render=False, learning_rate=LEARNING_RATE_DEFAULT,
                 batch_size=BATCH_SIZE_DEFAULT, render_mod=RENDER_MOD_DEFAULT, optimizer=INIT_OPT, directory=None):
        self.reward_sum = 0             # reward of an episode
        self.running_reward = None      # running reward of 100 games
        self.env = env                  # environment for OpenAI
        self.episode_num = 0            # number of episodes
        self.render = render            # if game should be rendered/shown
        self.render_mod = render_mod    # number of episodes before rendering

        self.nn_structure = num_layer_neurons
        self.nn_structure[0] = self.nn_structure[0] + self.nn_structure[-1]
        # setup policy gradient neural network
        self.poli_grad_nn = Policy_Gradient_NN(self.nn_structure,
                                               sess,
                                               learning_rate,
                                               batch_size,
                                               optimizer,
                                               directory=directory)

    """
    pong_eval
    brief: runs and learns pong game
    input: 
    output: 
    """

    def pong_eval(self):

        # reset environment and initialize vars and function calls
        observation = self.env.reset()
        self.reward_sum = 0
        fake_label = 0
        done = False
        episode_rewards, episode_observations, fake_labels = [], [], []
        prev_processed_observations = None
        test_nn = self.poli_grad_nn.test_nn
        train_eval = self.poli_grad_nn.train_nn

        # check if game is rendered/shown
        check = self.render and self.episode_num % self.render_mod == 0

        # play Pong
        while not done:
            if check:
                self.env.render()

            # preprocess observation before applying
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations)
            processed_observations = np.append(processed_observations, fake_label)
            # save input
            episode_observations.append(processed_observations)

            # determine action
            up_probability = test_nn([processed_observations])
            # action = choose_action(up_probability[0], self.chance_prob)
            action = choose_action(up_probability[0])

            # carry out the chosen action
            observation, reward, done, info = self.env.step(action)

            # save reward, decrease reward if AI loses, increase reward if AI wins
            self.reward_sum += reward
            if reward < 0:
                reward *= (1 - REWARD_CORRECTION)
            else:
                reward *= (1 + REWARD_CORRECTION)
            episode_rewards.append(reward)

            # change action to 1 == 2/up and 0 == 3/down
            # see here: http://cs231n.github.io/neural-networks-2/#losses
            fake_label = 1 if action == 2 else 0
            fake_labels.append(fake_label)

            # Episode finished/ begin updates
            if reward != 0:
                # if done:
                train_eval(np.array(episode_rewards), episode_observations, fake_labels)
                episode_rewards, episode_observations, fake_labels = [], [], []

        # update running reward
        self.running_reward = self.reward_sum if self.running_reward is None \
            else self.running_reward * 0.99 + self.reward_sum * 0.01

        # if AI wins, save network
        if self.reward_sum > 0:
            print("\n\n<<<<<<<<< I WON >>>>>>>>>\n\n")
            self.poli_grad_nn.saver.save(self.poli_grad_nn.sess, os.getcwd() + "/tmp/NN/")

        # update episode count
        self.episode_num += 1

        return

"""
/**** Pong_CNN_PoliGrad_Agent ****/
By: Edward Guevara 
References:  Sources
Description: CNN Agent that learns Pong using Policy Gradient optimization
"""
class Pong_CNN_PoliGrad_Agent(object):
    """Constructor for Pong_PoliGrad_Agent"""
    def __init__(self, height, width, num_layer_neurons, sess, env, render=False,
                 learning_rate=LEARNING_RATE_DEFAULT, batch_size=BATCH_SIZE_DEFAULT,
                 render_mod=RENDER_MOD_DEFAULT, optimizer=INIT_OPT, directory=None):
        self.reward_sum = 0             # reward of an episode
        self.running_reward = None      # running reward of 100 games
        self.env = env                  # environment for OpenAI
        self.episode_num = 0            # number of episodes
        self.render = render            # if game should be rendered/shown
        self.render_mod = render_mod    # number of episodes before rendering

        # setup policy gradient neural network
        self.poli_grad_cnn = Policy_Gradient_CNN(num_layer_neurons,
                                                height,
                                                width,
                                                sess,
                                                learning_rate,
                                                batch_size,
                                                optimizer,
                                                directory=directory)

    """
    pong_eval
    brief: runs and learns pong game
    input: 
    output: 
    """

    def pong_eval(self):

        # reset environment and initialize vars and function calls
        observation = self.env.reset()
        self.reward_sum = 0
        done = False
        episode_rewards, episode_observations, fake_labels = [], [], []
        prev_processed_observations = None
        test_nn = self.poli_grad_cnn.test_nn
        train_eval = self.poli_grad_cnn.train_nn

        # check if game is rendered/shown
        check = self.render and self.episode_num % self.render_mod == 0

        # play Pong
        while not done:
            if check:
                self.env.render()

            # preprocess observation before applying
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations)
            # save input
            episode_observations.append(processed_observations)

            # determine action
            up_probability = test_nn([processed_observations])
            # action = choose_action(up_probability[0], self.chance_prob)
            action = choose_action(up_probability[0])

            # carry out the chosen action
            observation, reward, done, info = self.env.step(action)

            # save reward, decrease reward if AI loses, increase reward if AI wins
            self.reward_sum += reward
            if reward < 0:
                reward *= (1 - REWARD_CORRECTION)
            else:
                reward *= (1 + REWARD_CORRECTION)
            episode_rewards.append(reward)

            # change action to 1 == 2/up and 0 == 3/down
            # see here: http://cs231n.github.io/neural-networks-2/#losses
            fake_label = 1 if action == 2 else 0
            fake_labels.append(fake_label)

            # Episode finished/ begin updates
            if reward != 0:
                # if done:
                train_eval(np.array(episode_rewards), episode_observations, fake_labels)
                episode_rewards, episode_observations, fake_labels = [], [], []

        # update running reward
        self.running_reward = self.reward_sum if self.running_reward is None \
            else self.running_reward * 0.99 + self.reward_sum * 0.01

        # if AI wins, save network
        if self.reward_sum > 0:
            print("\n\n<<<<<<<<< I WON >>>>>>>>>\n\n")
            self.poli_grad_cnn.saver.save(self.poli_grad_cnn.sess, os.getcwd() + "/tmp/NN/")

        # update episode count
        self.episode_num += 1

        return
