
from Agents.poli_grad_nn import *

class ConvolutionNN(PoliGradAgent):
    def __init__(self, num_layer_neurons, height, width, sess):
        self.layers = []
        self.x = tf.placeholder(tf.float32, [None, height*width])
        self.x_shaped = tf.reshape(self.x, [-1, height, width, 1])
        self.layer1 = self.create_new_conv_layer(self.x_shaped, 1, 32, [5, 5], [8, 8], [8, 8], name='layer1')
        self.layer2 = self.create_new_conv_layer(self.layer1, 32, 64, [5, 5], [8, 8], [8, 8], name='layer2')
        self.input_nn = tf.reshape(self.layer2, [-1, tf.shape(self.layer2)[0]])
        super().__init__(num_layer_neurons, sess, input_init=True)

        self.input_dimensions = height * width

    """Model functions for CNN (TensorFlow)"""
    # Input Layer
    def create_new_conv_layer(self,input_data, num_input_channels, num_filters, filter_shape,
                              pool_shape, stride_size, name):
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                           num_filters]

        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                              name=name + '_W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)

        # now perform max pooling
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, stride_size[0], stride_size[1], 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                                   padding='SAME')

        return out_layer

    """
    test_nn
    brief: tests the neural network
    input: input
    output:answer, accuracy 
    """

    def test_nn(self, input):
        return self.sess.run(self.layers[-1]['Qout'], feed_dict={self.x: input})

    """
    train_nn
    brief: calls model to train the neural network using policy gradient update
    input: input, correct_output
    output:last weight (for debugging) 
    """

    def train_nn(self, rewards, observations, actions):
        # discount the rewards for every action taken
        discount_expected = self.discount_rewards(rewards, GAMMA)

        # make feed dict and get gradients of network
        feed_dict = {self.reward_holder: discount_expected,
                     self.action_holder: actions, self.x: np.vstack(observations)}
        grads = self.sess.run(self.gradients, feed_dict=feed_dict)

        # add gradients to gradient buffers
        for idx, grad in enumerate(grads):
            self.gradBuffer[idx] += grad

        fake_labels, episode_observations, episode_up_prob, episode_rewards = [], [], [], []
        self.buffer_update += 1

        # apply the gradient when we hit the batch size
        if self.buffer_update % self.batch_size == 0:
            feed_dict = dictionary = dict(zip(self.gradient_holders, self.gradBuffer))
            _ = self.sess.run(self.update_batch, feed_dict=feed_dict)

            # reset gradient buffer
            for ix, grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad * 0
        return