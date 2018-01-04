import numpy as np
import tensorflow as tf
import gym

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]


def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]


def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image


def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195]  # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1  # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations

def choose_action(probability):
    random_value = np.random.uniform()
    # random_value = 0.5
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
        # signifies down in openai gym
        return 3
env = gym.make('Pong-v0')

# hyperparameters
H = 200 # number of hidden layer neurons
learning_rate = 1e-6
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

real_bs = 5 # Batch size when learning from real environment

tf.reset_default_graph()
observations = tf.placeholder(tf.float32, [None,6400] , name="input_x")
W1 = tf.get_variable("W1", shape=[6400, H],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_sum(loglik * advantages)
newGrads = tf.gradients(loss,tvars)
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs,drs,ys,ds = [],[],[],[]
running_reward = None
prev_running_reward = 0
reward_sum = 0
batch_total_reward = 0
episode_number = 1
real_episodes = 1
init = tf.global_variables_initializer()
batch_size = real_bs
done = False

prev_processed_observations = None

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()
    x = observation

    gradBuffer = sess.run(tvars)
    gradBuffer = resetGradBuffer(gradBuffer)

    while True:
        while not done:
            # Start displaying environment once performance is acceptably high.
            if real_episodes % 20 == 0:
                env.render()
            processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                          prev_processed_observations,
                                                                                          6400)
            x = np.reshape(processed_observations, [1, 6400])

            tfprob = sess.run(probability, feed_dict={observations: x})
            action = choose_action(tfprob)

            # record various intermediates (needed later for backprop)
            xs.append(x)
            y = 1 if action == 2 else 0
            ys.append(y)

            observation, reward, done, info = env.step(action)

            reward_sum += reward

            ds.append(done * 1)
            drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

            # if reward != 0:
            if done:
                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                epd = np.vstack(ds)
                xs, drs, ys, ds = [], [], [], []  # reset array memory

                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = sess.run(newGrads, feed_dict={observations: epx,
                                                      input_y: epy,
                                                      advantages: discounted_epr})
                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

                episode_number += 1

                if episode_number % real_bs == 0:
                    sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0],
                                                     W2Grad: gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)
                    episode_number = 0

        real_episodes += 1
        batch_total_reward += reward_sum
        done = False
        print('Episode Reward: %f' % reward_sum)
        if reward_sum >= 0:
            print('<<<<< I WON >>>>>')

        if real_episodes % real_bs == 0:
            running_reward = batch_total_reward if running_reward is None \
                else running_reward * 0.99 + batch_total_reward * 0.01
            print(
            'World Perf: Episode %f. Reward: %f. action: %f. mean reward: %f. diff: %f' % (
            real_episodes, batch_total_reward / real_bs,
            action, running_reward / real_bs,
            running_reward / real_bs - prev_running_reward))
            prev_running_reward = running_reward / real_bs
            batch_total_reward = 0
        reward_sum = 0
        observation = env.reset()