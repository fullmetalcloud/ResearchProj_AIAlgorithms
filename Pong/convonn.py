
import gym
import numpy as np
import tensorflow as tf

from neuralnet import *

class ConvolutionNN(NeuralNetwork):
    def __init__(self, num_hidden_layer_neurons, height, width, gamma=0.99,
                 decay_rate=0.99, learning_rate=1e-4, batch_size=10):
        self.height = height
        self.width = width
        self.image_dimensions = height * width
        #TODO change 64 into determined value based on flattened output array
        super().__init__(num_hidden_layer_neurons,64,gamma,decay_rate,learning_rate,batch_size)
        self.x = tf.placeholder(tf.float32, [None, height * width])
        self.x_shaped = tf.reshape(self.x, [-1, height, width, 1])
        self.layer1 = self.create_new_conv_layer(self.x_shaped, 1, 32, [5, 5], [8, 8], name='layer1')
        self.layer2 = self.create_new_conv_layer(self.layer1, 32, 64, [5, 5], [8, 8], name='layer2')
        self.flattened = tf.reshape(self.layer2, [-1, 64])

        self.init_op = tf.global_variables_initializer()

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

    def eval(self,env, render=False):
        observation = env.reset()
        self.reward_sum = 0
        self.running_reward = None
        done = False
        episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
        prev_processed_observations = None
        with tf.Session() as sess:
            sess.run(self.init_op)
            while not done:
                if render and self.episode_num % self.render_mod == 0:
                    env.render()
                processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                              prev_processed_observations,
                                                                                              self.image_dimensions)
                a = sess.run(self.flattened, feed_dict={self.x: [processed_observations]})[0]

                hidden_layer_values, up_probability = self.apply_neural_nets(a)

                episode_observations.append(a)
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
