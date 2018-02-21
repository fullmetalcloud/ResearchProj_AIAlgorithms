"""
references for optimization explanation and implementation
# https://medium.com/all-of-us-are-belong-to-machines/gentlest-introduction-to-tensorflow-part-2-ed2a0a7a624f
# https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
# http://ruder.io/optimizing-gradient-descent/index.html
# https://stats.stackexchange.com/questions/266968/how-does-minibatch-gradient-descent-update-the-weights-for-each-example-in-a-bat/266974
"""

import tensorflow as tf
import numpy as np
from parameters import *
"""
policy_gradient_optimizer
brief: initializes policy gradient optimization
input: input, optimizer
output: reward_holder, action_holder, entropy 
"""
def policy_gradient_optimizer(output, output_size):

    # IMPLEMENTATION AND SAMPLE: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb

    # The next six lines establish the training procedure. We feed the reward and chosen action into the network
    # to compute the loss, and use it to update the network.
    reward_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='reward_holder')

    # Determine responsible outputs of each action
    if 1 < output_size:
        # if there are more than 2 outputs then gather the chosen output nodes
        action_holder = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='action_holder')
        indexes = tf.range(0, tf.shape(output)[0]) * tf.shape(output)[1] + action_holder
        responsible_outputs = tf.gather(tf.reshape(output, [-1]), indexes)

    else:
        # if one output, determine if action is 0 or 1 from action_holder and get the likelihood of action based
        # on output (1 => output, 0 => 1 - output)
        action_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='action_holder')
        # self.responsible_outputs = self.action_holder * (self.action_holder - self.output) \
        #                    + (1 - self.action_holder) * (self.action_holder + self.output)
        responsible_outputs = action_holder * (output) \
                                   + (1 - action_holder) * (1 - output)

    # compute loss for all actions taken in episode
    # self.entropy = tf.sqrt(self.responsible_outputs ** 2)
    entropy = tf.log(responsible_outputs)

    # if loss is smaller than e-50 (smallest value of float32, mask as 0)
    entropy = tf.where(tf.is_nan(entropy), tf.zeros_like(entropy), entropy)
    entropy = tf.where(tf.is_inf(entropy), tf.zeros_like(entropy), entropy)
    
    # loss calculated based on determined actions and adjusted rewards
    loss = -tf.reduce_mean(entropy * reward_holder)

    return reward_holder, action_holder, entropy, loss

"""Policy Gradient - Reward Adjustment Function"""
def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action 
    you took a step ago. This implements that logic by discounting the reward on previous actions based on how 
    long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

"""
stochastic_grad_desc
brief: initializes stochastic gradient descent optimization
input: 
output:gradient_holders, gradients, update_batch, init_op, gradBuffer, saver 
"""
def stochastic_grad_desc(loss, optimizer, sess, learning_rate):

    # initialize trainable vars and gradients
    tvars = tf.trainable_variables()
    gradient_holders = []

    # placeholders for gradient buffers to update NN
    for idx, var in enumerate(tvars):
        placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
        gradient_holders.append(placeholder)

    gradients = tf.gradients(loss, tvars)

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
    clipped_grad, _ = tf.clip_by_global_norm(gradient_holders, CLIPPING_DEFAULT)
    update_batch = train_step.apply_gradients(zip(clipped_grad, tvars))

    # Tensorflow Saver initialization
    saver = tf.train.Saver()

    # tensorflow initialize variables
    init_op = tf.global_variables_initializer()

    # setup session for game
    sess.run(init_op)

    # setup gradient buffer
    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    return gradient_holders, gradients, update_batch, init_op, gradBuffer, saver

"""
default_optimizer
brief: initializes single optimizer
input: output, optimizer
output: 
"""
def default_optimizer(loss, optimizer, sess):

    # training optimization
    if optimizer == 'Adam':
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_DEFAULT)
    elif optimizer == 'RMSProp':
        train_step = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE_DEFAULT, decay=DECAY_RATE)
    elif optimizer == 'Adagrad':
        train_step = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE_DEFAULT)
    else:
        train_step = None
        print("NEED train_step DEFINED")
        exit(1)

    # set up optimizer minimize (gradient and gradient update in one step)
    update_batch = optimizer.minimize(loss)

    # tensorflow initialize variables
    init_op = tf.global_variables_initializer()

    # setup session for game
    sess.run(init_op)

    return update_batch, init_op