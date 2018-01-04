import os

from Agents.nn import NeuralNetwork
import numpy as np
from pong_image import *
import tensorflow as tf
from Agents.parameters import *


# POLICY GRADIENT
# change decision rewards based on gaining a point (all actions leading to this has +1 reward to encourage)
# or losing a point (all actions leading to this has -1 reward to discourage)
# Also, decrease reward multiplier the farther the action is from end of episode
# EXPLANATION: http://karpathy.github.io/2016/05/31/rl/ under training protocol
# IMPLEMENTATION AND SAMPLE: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb

"""
/**** PoliGradAgent ****/
By: Edward Guevara 
References:  Sources
Description: 
"""
class PoliGradAgent(NeuralNetwork):
    """Constructor for PolicyGradientAgent"""

    def __init__(self, num_layer_neurons, sess, learning_rate=LEARNING_RATE_DEFAULT, batch_size=BATCH_SIZE_DEFAULT,
                 render_mod=RENDER_MOD_DEFAULT, input_init=False, optimizer=INIT_OPT):

        self.sess = sess                    # session for TensorFlow
        self.batch_size = batch_size        # number of batches before learning
        self.render_mod = render_mod        # number of episodes before rendering
        self.buffer_update = 0              # increment to determine when to update NN w/ gradients
        self.learning_rate = learning_rate  # learning rate for optimization
        self.reward_sum = 0                 # reward sum after every episode
        self.episode_num = 0                # episode number
        self.chance_prob = RAND_ACTION_PROB # probability to use random action (multiple outputs only)
        self.batch_loss = []
        self.batch_entropy = []

        # initialize Neural Network structure
        super().__init__(num_layer_neurons, input_init=input_init)

        # initialize last layer
        self.output = self.layers[-1]['Qout']

        # IMPLEMENTATION AND SAMPLE: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb

        # The next six lines establish the training procedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='reward_holder')

        # Determine responsible outputs of each action
        if 1 < num_layer_neurons[-1]:
            # if there are more than 2 outputs then gather the chosen output nodes
            self.action_holder = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='action_holder')
            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
            # self.responsible_outputs = tf.subtract(tf.ones(tf.shape(self.responsible_outputs)),
            #                                 self.responsible_outputs)


        else:
            # if one output, determine if action is 0 or 1 from action_holder and get the likelihood of action based
            # on output (1 => output, 0 => 1 - output)
            self.action_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='action_holder')
            # self.responsible_outputs = self.action_holder * (self.action_holder - self.output) \
            #                    + (1 - self.action_holder) * (self.action_holder + self.output)
            self.responsible_outputs = self.action_holder * (self.output) \
                                   + (1 - self.action_holder) * (1 - self.output)

        # compute loss for all actions taken in episode
        # self.entropy = tf.sqrt(self.responsible_outputs ** 2)
        self.entropy = tf.log(self.responsible_outputs)

        # loss calculated based on determined actions and adjusted rewards
        self.loss = -tf.reduce_mean(self.entropy * self.reward_holder)

        # initialize trainable vars and gradients
        tvars = tf.trainable_variables()
        self.gradient_holders = []

        # placeholders for gradient buffers to update NN
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        # training optimization
        if optimizer == 'Adam':
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'RMSProp':
            train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=DECAY_RATE)
        elif optimizer == 'Adagrad':
            train_step = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        else:
            train_step = None
            print("NEED train_step DEFINED")
            exit(1)
        self.update_batch = train_step.apply_gradients(zip(self.gradient_holders, tvars))

        # Tensorflow Saver initialization
        self.saver = tf.train.Saver()

        # tensorflow initialize variables
        self.init_op = tf.global_variables_initializer()

        # setup session for game
        self.sess.run(self.init_op)

        # setup gradient buffer
        self.gradBuffer = self.sess.run(tf.trainable_variables())
        for ix, grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0

    """Choose Action for OpenAI Pong"""

    # # chooses action based on double output
    # def choose_action(self, action):
    #     random_value = np.random.uniform()
    #     # print(action)
    #     # random_value = 0.5
    #     if random_value < self.chance_prob:
    #         return np.random.randint(2,4)
    #     else:
    #         if action[0] >= action[1]:
    #             # signifies down in openai gym
    #             return 3
    #         else:
    #             # signifies up in openai gym
    #             return 2

    # chooses action based on single output

    def choose_action(self, action):
        random_value = np.random.uniform()
        if random_value < action:
            # signifies up in openai gym
            return 2
        else:
            # signifies down in openai gym
            return 3

    """Policy Gradient - Reward Adjustment Function"""
    def discount_rewards(self, rewards, gamma):
        """ Actions you took 20 steps before the end result are less important to the overall result than an action 
        you took a step ago. This implements that logic by discounting the reward on previous actions based on how 
        long ago they were taken"""
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            # if rewards[t] != 0:
            #     running_add = 0
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    """
    train_nn
    brief: calls model to train the neural network using policy gradient update
    input: input, correct_output
    output:last weight (for debugging) 
    """

    def train_nn(self, rewards, observations, actions):
        # discount the rewards for every action taken
        discount_expected = self.discount_rewards(rewards, GAMMA)
        discount_expected -= np.mean(discount_expected)
        discount_expected /= np.std(discount_expected)

        # make feed dict and get gradients of network
        feed_dict = {self.reward_holder: np.vstack(discount_expected),
                     self.action_holder: np.vstack(actions),
                     self.input_nn: np.vstack(observations)}
        grads, loss, o = self.sess.run([self.gradients, self.loss,self.entropy], feed_dict=feed_dict)
        if np.isnan(loss):
            print('\n\n <<<NaN OCCURRED>>>\n\n')
            self.sess.run(self.init_op)
        self.batch_loss.append(loss)
        self.batch_entropy.append(np.mean(o))

        # add gradients to gradient buffers
        for idx, grad in enumerate(grads):
            self.gradBuffer[idx] += grad
            # print(self.gradBuffer[idx])

        self.buffer_update += 1

        # apply the gradient when we hit the batch size
        if self.buffer_update % self.batch_size == 0:
            feed_dict = dict(zip(self.gradient_holders, self.gradBuffer))
            _ = self.sess.run(self.update_batch, feed_dict=feed_dict)

            print(sum(self.batch_loss)/len(self.batch_loss))
            print(sum(self.batch_entropy)/len(self.batch_entropy))
            self.batch_entropy, self.batch_loss = [], []
            # reset gradient buffer
            for ix, grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad * 0

            # Reduce random action probability
            if self.chance_prob <= RAND_LOWEST_PROB:
                self.chance_prob = RAND_LOWEST_PROB
            else:
                self.chance_prob /= RAND_REDUCTION_DIV
        return

    """
    test_nn
    brief: tests the neural network
    input: input
    output:answer, accuracy 
    """

    def test_nn(self, input):
        return self.sess.run(self.output, feed_dict={self.input_nn: input})

    # evaluation function
    def eval(self, env, render=False):

        # reset environment and initialize vars and function calls
        observation = env.reset()
        self.reward_sum = 0
        done = False
        episode_rewards, episode_observations, fake_labels = [], [], []
        prev_processed_observations = None
        test_eval = self.test_nn
        train_eval = self.train_nn
        choose_action = self.choose_action

        # check if game is rendered/shown
        check = render and self.episode_num % self.render_mod == 0

        # play Pong
        while not done:
            if check:
                env.render()

            # preprocess observation before applying
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations,
                                                                                          self.input_dimensions)
            # save input
            episode_observations.append(processed_observations)

            # determine action
            up_probability = test_eval([processed_observations])
            # action = choose_action(up_probability[0], self.chance_prob)
            action = choose_action(up_probability[0])


            # carry out the chosen action
            observation, reward, done, info = env.step(action)

            # save reward, decrease reward if AI loses, increase reward if AI wins
            self.reward_sum += reward
            # if reward < 0: reward *= (1 - REWARD_CORRECTION)
            # else: reward *= (1 + REWARD_CORRECTION)
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

        # if AI wins, save network
        if self.reward_sum > 0:
            print("\n\n<<<<<<<<< I WON >>>>>>>>>\n\n")
            self.saver.save(self.sess, os.getcwd() + "/tmp/NN/")

        # update episode count
        self.episode_num += 1

        return

    """
    test_eval
    brief: tests evaluation of class using pole balancing
    input: env
    output:accuracy, num episodes 
    """
    def test_eval(self, env):
        choose_action = self.choose_action
        test_acc = False
        observation = env.reset()
        chance_prob = 0.1
        accuracy = 0
        count = 0
        num_episodes = 0
        desired_acc = 200
        avg_reward = 0
        highest_reward = 0
        episode_rewards, episode_observations, episode_up_prob, actions = [], [], [], []
        self.sess.run(self.init_op)
        while not test_acc:
            # if num_episodes % 5000 == 0:
            #     self.sess.run(self.init_op)
            #     chance_prob = 0.5

            # render episode
            if num_episodes % 100 == 0:
                env.render()

            # reshape and record observation
            observation = np.reshape(observation, [1, 4])
            episode_observations.append(observation)

            # run NN and get left or right for action
            up_probability = self.test_nn(observation)[0]

            action = choose_action(up_probability[0])
            action = 1 if action == 2 else 0
            observation, reward, done, info = env.step(action)

            # record action taken
            actions.append(action)

            # sum and record reward
            self.reward_sum += reward
            episode_rewards.append(reward)

            # when episode finishes
            if done:
                # print("reward: %f, numEpisodes: %i" % (self.reward_sum, num_episodes))

                # train the NN
                self.train_nn(np.array(episode_rewards), episode_observations, actions)

                episode_rewards, episode_observations, episode_up_prob, actions = [], [], [], []

                # check if pole lasted long enough
                if desired_acc <= self.reward_sum:
                    test_acc = True
                else:
                    avg_reward += self.reward_sum
                    highest_reward = self.reward_sum if self.reward_sum > highest_reward else highest_reward
                    if num_episodes % 100 == 0:
                        print("numEpisodes: %i" % (num_episodes))
                        print("Average Reward: %f" % (avg_reward/100))
                        print("Reward: %i" % (highest_reward))
                        highest_reward = 0
                        avg_reward = 0
                    self.reward_sum = 0
                    observation = env.reset()
                    num_episodes += 1
        return self.reward_sum, num_episodes