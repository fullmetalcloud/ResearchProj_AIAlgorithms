
from Agents.poli_grad_nn import *

class ConvolutionNN(PoliGradAgent):
    def __init__(self, num_layer_neurons, height, width, sess):
        self.layers = []
        self.x = tf.placeholder(tf.float32, [None, height*width])
        self.x_shaped = tf.reshape(self.x, [-1, height, width, 1])
        self.layer1 = self.create_new_conv_layer(self.x_shaped, 1, 32,[4, 4], [4, 4], [8, 8], [8, 8], name='layer1')
        self.layer2 = self.create_new_conv_layer(self.layer1, 32, 64, [4, 4], [4, 4], [2, 2], [2, 2], name='layer2')
        self.input_nn = tf.reshape(self.layer2, [tf.shape(self.layer2)[0], -1])
        super().__init__(num_layer_neurons, sess, input_init=True)

        self.input_dimensions = height * width

    """Model functions for CNN (TensorFlow)"""
    # Input Layer
    def create_new_conv_layer(self,input_data, num_input_channels, num_filters, filter_shape, filter_stride_size,
                              pool_shape, stride_size, name):
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                           num_filters]

        # initialise weights and bias for the filter
        xavier_var = 1/np.sqrt(filter_shape[0]*filter_shape[1]*num_input_channels*num_filters)
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape,
                                                  stddev= xavier_var),
                              name=name + '_W')
        bias = tf.Variable(tf.zeros([num_filters]), name=name + '_b')

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data,
                                 weights,
                                 [1, filter_stride_size[0], filter_stride_size[1], 1],
                                 padding='SAME')

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
        discount_expected -= np.mean(discount_expected)
        discount_expected /= np.std(discount_expected)

        # make feed dict and get gradients of network
        feed_dict = {self.reward_holder: np.vstack(discount_expected),
                     self.action_holder: np.vstack(actions),
                     self.x: np.vstack(observations)}
        grads, loss, o = self.sess.run([self.gradients, self.loss, self.entropy], feed_dict=feed_dict)
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

            print(sum(self.batch_loss) / len(self.batch_loss))
            print(sum(self.batch_entropy) / len(self.batch_entropy))
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