
import os
import time

import gym
import tensorflow as tf

from pong_agents import Pong_CNN_PoliGrad_Agent

def cnn_main():
    # for testing
    render = False

    #hyperparameters
    num_layer_neurons_ConvoNN = [128, 200, 1]
    height = 80
    width = 80
    reward = []

    # location of saved files (networks, speciation graphs, videos)
    directory = os.getcwd() + "/tmp/ConvoNN"

    #parameters for monitoring and recording games
    running_reward_convo_NN = None
    prev_running_reward_convo_NN = 0
    best_reward_convo_NN = None
    episode_number_ConvoNN = 0
    time_convonn, reward_convonn = [0], [0]
    record_rate_convoNN = 100

    # tensorflow session init
    sess_convonn = tf.Session()

    # openAI env init
    envConvoNN = gym.make('Pong-v0')

    # openAI env recording videos init
    envConvoNN = gym.wrappers.Monitor(envConvoNN,
                                      directory,
                                      force=True,
                                      video_callable=lambda
                                          episode_id: 0 == episode_number_ConvoNN % record_rate_convoNN)

    # file creation of Convolutional NN
    fileconvoNN = open(directory + '/ConvoNNInfo', 'w')
    fileconvoNN.write('episode,time,score,avg_loss,avg_entropy\n')

    # class creation of Convolutional NN Agent
    convo_nn = Pong_CNN_PoliGrad_Agent(height,
                                       width,
                                       num_layer_neurons_ConvoNN,
                                       sess_convonn,
                                       envConvoNN,
                                       render,
                                       directory=directory)

    try:
        while True:
            # evaluate Convolutional Fully Connected Neural Network
            episode_number_ConvoNN = convo_nn.episode_num
            print(episode_number_ConvoNN)
            start = time.time()
            convo_nn.pong_eval()
            end = time.time()
            print("Convo NN Reward: %i Time: %.3f" % (convo_nn.reward_sum, (end - start)))
            fileconvoNN.write('%i,%.3f,%i\n' % (episode_number_ConvoNN, (end - start), convo_nn.reward_sum))
            fileconvoNN.flush()
            # score and improvement records
            if best_reward_convo_NN is None:
                best_reward_convo_NN = convo_nn.reward_sum
            elif convo_nn.reward_sum > best_reward_convo_NN:
                best_reward_convo_NN = convo_nn.reward_sum

            if episode_number_ConvoNN % convo_nn.render_mod == 0:
                print(
                    'CNN World Perf: Episode %i. running reward: %f. diff: %f time: %.4f Top Score: %i' % (
                        convo_nn.episode_num,
                        convo_nn.running_reward,
                        convo_nn.running_reward - prev_running_reward_convo_NN,
                        sum(time_convonn) / len(time_convonn),
                        best_reward_convo_NN))
                prev_running_reward_convo_NN = convo_nn.running_reward
                time_convonn, reward_convonn = [], []
            else:
                time_convonn.append(end - start)
                reward_convonn.append(convo_nn.reward_sum)
            # if convo_nn.episode_num == 1000:
                # record_rate_convoNN = 1000
                # convo_nn.render_mod = 1000
            # elif convo_nn.episode_num == 10000:
                # record_rate_convoNN = 2000
                # convo_nn.render_mod = 2000
    except KeyboardInterrupt:
        fileconvoNN.close()

cnn_main()