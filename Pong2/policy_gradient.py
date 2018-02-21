
from nn import Neural_Network
from cnn import Convolutional_NN
from optimizer import *

"""
/**** Policy_Gradient_NN ****/
By: Edward Guevara 
References:  Sources
Description: 
"""

class Policy_Gradient_NN(object):
    """Constructor for PolicyGradientAgent"""

    def __init__(self, num_layer_neurons, sess, learning_rate=LEARNING_RATE_DEFAULT,
                 batch_size=BATCH_SIZE_DEFAULT, optimizer=INIT_OPT, directory=None):
        self.sess = sess  # session for TensorFlow
        self.batch_size = batch_size  # number of batches before learning
        self.buffer_update = 0  # increment to determine when to update NN w/ gradients
        self.learning_rate = learning_rate  # learning rate for optimization
        self.batch_loss = []  # save losses in a batch (debugging)
        self.batch_entropy = [] # save entropy of a batch (debugging)
        if directory is not None:
            self.batch_file = open(directory + "/BatchUpdates", 'w') # file name for recording info
            self.batch_file.write("batch,entropy,loss\n")
            self.batch_file.flush()
        else:
            self.batch_file = None


        # setup neural network
        self.neural_net = Neural_Network(num_layer_neurons)

        # get output of neural network
        self.output = self.neural_net.layers[-1]['Qout']

        # setup policy gradient optimization
        policy_info = policy_gradient_optimizer(self.output, num_layer_neurons[-1])

        self.reward_holder = policy_info[0]     # tensor to input saved rewards
        self.action_holder = policy_info[1]     # tensor to store actions taken
        self.entropy = policy_info[2]           # entropy of gradient update
        self.loss = policy_info[3]              # loss after gradient update

        # setup minibatch optimization
        self.gradient_holders, self.gradients, self.update_batch,  self.init_op, \
        self.gradBuffer, self.saver = stochastic_grad_desc(self.loss, optimizer, self.sess, learning_rate)


    """
    test_nn
    brief: tests the neural network
    input: input
    output:answer, accuracy 
    """

    def test_nn(self, input):
        feed_dict = {self.neural_net.input_nn: input, self.neural_net.dropout: self.neural_net.dropout_percent}
        return self.sess.run(self.output, feed_dict=feed_dict)

    """
    train_nn
    brief: calls model to train the neural network using policy gradient update
    input: input, correct_output
    output:last weight (for debugging) 
    """

    def train_nn(self, rewards, observations, actions):
        # discount the rewards for every action taken
        discount_expected = discount_rewards(rewards, GAMMA)
        discount_expected -= np.mean(discount_expected)
        discount_expected /= np.std(discount_expected)

        # make feed dict and get gradients of network
        feed_dict = {self.reward_holder: np.vstack(discount_expected),
                     self.action_holder: np.vstack(actions),
                     self.neural_net.input_nn: np.vstack(observations),
                     self.neural_net.dropout: self.neural_net.dropout_percent}
        grads, loss, o = self.sess.run([self.gradients, self.loss, self.entropy], feed_dict=feed_dict)
        if np.isnan(loss):
            print('\n\n <<<NaN OCCURRED>>>\n\n')
            self.sess.run(self.init_op)
        self.batch_loss.append(loss)
        self.batch_entropy.append(np.mean(o))

        if self.batch_file is not None:
            self.batch_file.write("%i,%.8f,%.8f\n" % (self.buffer_update, loss, np.mean(o)))
            self.batch_file.flush()
        # add gradients to gradient buffers
        for idx, grad in enumerate(grads):
            self.gradBuffer[idx] += grad

        self.buffer_update += 1

        # apply the gradient when we hit the batch size
        if self.buffer_update % self.batch_size == 0:

            feed_dict = dict(zip(self.gradient_holders, self.gradBuffer))
            _ = self.sess.run(self.update_batch, feed_dict=feed_dict)

            # print(sum(self.batch_loss)/len(self.batch_loss))
            print(sum(self.batch_entropy)/len(self.batch_entropy))
            self.batch_entropy, self.batch_loss = [], []
            # reset gradient buffer
            for ix, grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad * 0

            self.neural_net.dropout_percent *= 1.1
            self.neural_net.dropout_percent = 1.0 if self.neural_net.dropout_percent > 1.0 \
                else self.neural_net.dropout_percent


"""
/**** Policy_Gradient_CNN ****/
By: Edward Guevara 
References:  Sources
Description: 
"""

class Policy_Gradient_CNN(object):
    """Constructor for Policy_Gradient_CNN"""

    def __init__(self, hidden_layer_neurons, height, width, sess, learning_rate=LEARNING_RATE_DEFAULT,
                 batch_size=BATCH_SIZE_DEFAULT, optimizer=INIT_OPT, directory=None):
        self.sess = sess  # session for TensorFlow
        self.batch_size = batch_size  # number of batches before learning
        self.buffer_update = 0  # increment to determine when to update NN w/ gradients
        self.learning_rate = learning_rate  # learning rate for optimization
        self.batch_loss = []  # save losses in a batch (debugging)
        self.batch_entropy = [] # save entropy of a batch (debugging)
        if directory is not None:
            self.batch_file = open(directory + "/batch_updates", 'w') # file name for recording info
            self.batch_file.write("batch,entropy,loss\n")
            self.batch_file.flush()
        else:
            self.batch_file = None

        # setup neural network
        self.cnn = Convolutional_NN(height, width, hidden_layer_neurons)
        self.neural_net = self.cnn.neural_net

        # get output of neural network
        self.output = self.neural_net.layers[-1]['Qout']

        # setup policy gradient optimization
        policy_info = policy_gradient_optimizer(self.output, hidden_layer_neurons[-1])

        self.reward_holder = policy_info[0]     # tensor to input saved rewards
        self.action_holder = policy_info[1]     # tensor to store actions taken
        self.entropy = policy_info[2]           # entropy of gradient update
        self.loss = policy_info[3]              # loss after gradient update

        # setup minibatch optimization
        self.gradient_holders, self.gradients, self.update_batch,  self.init_op, \
        self.gradBuffer, self.saver = stochastic_grad_desc(self.loss, optimizer, self.sess, learning_rate)

    """
    test_nn
    brief: tests the neural network
    input: input
    output:answer, accuracy 
    """

    def test_nn(self, input):
        feed_dict = {self.cnn.input_nn: input, self.cnn.dropout: self.cnn.dropout_percent}
        return self.sess.run(self.output, feed_dict=feed_dict)
    """
    train_nn
    brief: calls model to train the neural network using policy gradient update
    input: input, correct_output
    output:last weight (for debugging) 
    """

    def train_nn(self, rewards, observations, actions, directory=None):
        # discount the rewards for every action taken
        discount_expected = discount_rewards(rewards, GAMMA)
        discount_expected -= np.mean(discount_expected)
        discount_expected /= np.std(discount_expected)

        # make feed dict and get gradients of network
        feed_dict = {self.reward_holder: np.vstack(discount_expected),
                     self.action_holder: np.vstack(actions),
                     self.cnn.input_nn: np.vstack(observations),
                     self.cnn.dropout: self.cnn.dropout_percent}
        grads, loss, o = self.sess.run([self.gradients, self.loss, self.entropy], feed_dict=feed_dict)
        if np.isnan(loss):
            print('\n\n <<<NaN OCCURRED>>>\n\n')
            self.sess.run(self.init_op)
        self.batch_loss.append(loss)
        self.batch_entropy.append(np.mean(o))

        if self.batch_file is not None:
            self.batch_file.write("%i,%.8f,%.8f\n" % (self.buffer_update, loss, np.mean(o)))
            self.batch_file.flush()

        # add gradients to gradient buffers
        for idx, grad in enumerate(grads):
            self.gradBuffer[idx] += grad
            # print(self.gradBuffer[idx])

        self.buffer_update += 1

        # apply the gradient when we hit the batch size
        if self.buffer_update % self.batch_size == 0:

            feed_dict = dict(zip(self.gradient_holders, self.gradBuffer))
            _ = self.sess.run(self.update_batch, feed_dict=feed_dict)

            # print(sum(self.batch_loss)/len(self.batch_loss))
            print(sum(self.batch_entropy)/len(self.batch_entropy))
            self.batch_entropy, self.batch_loss = [], []
            # reset gradient buffer
            for ix, grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad * 0

            self.cnn.dropout_percent *= DROPOUT_INCREASE_PERCENT
            self.cnn.dropout_percent = 1.0 if self.cnn.dropout_percent > 1.0 \
                else self.cnn.dropout_percent

