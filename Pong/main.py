import os
import time
from multiprocessing import Manager

import gym
import matplotlib.pyplot as pyplot
import numpy as np
import tensorflow as tf

from NEAT.neatq import NEATPong, NeatTest
from NEAT.plot import *
from Agents.poli_grad_nn import PoliGradAgent
from Agents.convo_nn import ConvolutionNN
from unit_tests import *


def main():
    # for testing
    render = True

    Run_Pong = True
    neuralNet = True
    convoNN = False
    NeatAlgo = False

    NNTests = False
    NEATTests = False
    PoliGradNNTests = False

    render_mod = 10

    # hyperparameters
    numInputs = 6400
    numOutputs = 1
    num_layer_neurons_NN = [6400, 200, 1]
    num_layer_neurons_ConvoNN = [256, 200, 1]
    height = 80
    width = 80
    reward = []

    # location of saved files (networks, speciation graphs, videos)
    directory = os.getcwd() + "/tmp/"

    #parameters for monitoring and recording games
    running_reward_NN = None
    prev_running_reward_NN = 0
    best_reward_NN = None
    episode_number_NN = 0
    time_nn, reward_nn  = [0], [0]


    episode_number_ConvoNN = 0
    time_convonn, reward_convonn = [0], [0]
    episode_number_NEAT = 0
    time_neat, reward_neat = [0], [0]
    record_rate_NN = 100
    record_rate_convoNN = 100
    record_rate_NEAT = 100

    sortedPop = []

    #initialize gym environment
    if neuralNet:
        # tensorflow session init
        sess_nn = tf.Session()

        # openAI env init
        envNN = gym.make('Pong-v0')

        # openAI env recording videos init
        envNN = gym.wrappers.Monitor(envNN,
                                     directory + "NN",
                                     force=True,
                                     video_callable=lambda episode_id: 0 == episode_number_NN % record_rate_NN)

        # file creation of NN
        fileNN = open(directory + 'NN/NNInfo', 'w')
        fileNN.write('episode,time,score\n')

        # class creation of NN Agent
        nn = PoliGradAgent(num_layer_neurons_NN,
                           sess_nn)

    if convoNN:
        # tensorflow session init
        sess_convonn = tf.Session()
        # openAI env init
        envConvoNN = gym.make('Pong-v0')

        # openAI env recording videos init
        envConvoNN = gym.wrappers.Monitor(envConvoNN,
                                          directory + "ConvoNN",
                                          force=True,
                                          video_callable=lambda
                                              episode_id: 0 == episode_number_ConvoNN % record_rate_convoNN)

        # file creation of Convolutional NN
        fileconvoNN = open(directory + 'ConvoNN/ConvoNNInfo', 'w')
        fileconvoNN.write('episode,time,score\n')

        # class creation of Convolutional NN Agent
        convo_nn = ConvolutionNN(num_layer_neurons_ConvoNN,
                                 height,
                                 width,
                                 sess_convonn)
    if NeatAlgo:

        # openAI env init
        envNEAT = gym.make('Pong-v0')

        # openAI env recording videos init
        envNEAT = gym.wrappers.Monitor(envNEAT,
                                       directory+"NEAT",
                                       force=True,
                                       video_callable=lambda episode_id: 0 == episode_number_NEAT % record_rate_NEAT)

        # multiprocessing manager for sharing variables
        manager = Manager()

        # class creation of NEAT Agent
        neat = NEATPong(numInputs,
                        numOutputs,
                        envNEAT,
                        render,
                        render_mod,
                        manager=manager)

    # Unit testing of classes (XOR Example)
    if NNTests:
        UnitTest_NNAgent()

    if PoliGradNNTests:
        UnitTest_PoliGradNN()

    if NEATTests:
        UnitTest_NEAT()

    if Run_Pong:
        try:
            while True:
                # Pong Game
                if neuralNet:
                    # evaluate Fully Connected Neural Network
                    episode_number_NN = nn.episode_num
                    print(episode_number_NN)
                    start = time.time()
                    nn.eval(envNN, render)
                    end = time.time()
                    print("NN Reward: %i Time: %.3f" % (nn.reward_sum, end-start))
                    fileNN.write('%i,%.3f,%i\n' % (episode_number_NN,
                                                         (end - start),
                                                         nn.reward_sum))

                    # score and improvement records
                    running_reward_NN = nn.reward_sum if running_reward_NN is None \
                        else running_reward_NN * 0.99 + nn.reward_sum * 0.01
                    if best_reward_NN is None:
                        best_reward_NN = nn.reward_sum
                    elif nn.reward_sum > best_reward_NN:
                        best_reward_NN = nn.reward_sum


                    if episode_number_NN % nn.batch_size == 0:

                        print(
                            'World Perf: Episode %f. mean reward: %f. diff: %f time: %.4f Top Score: %i' % (
                                nn.episode_num,
                                running_reward_NN,
                                running_reward_NN - prev_running_reward_NN,
                                sum(time_nn) / len(time_nn),
                                best_reward_NN))
                        prev_running_reward_NN = running_reward_NN
                        time_nn,reward_nn = [], []
                    else:
                        time_nn.append(end - start)
                        reward_nn.append(nn.reward_sum)
                    if episode_number_NN == 1000:
                        # record_rate_NN = 1000
                        nn.render_mod = 1000
                    elif episode_number_NN == 10000:
                        # record_rate_NN = 2000
                        nn.render_mod = 2000

                if convoNN:
                    # evaluate Convolutional Fully Connected Neural Network
                    episode_number_ConvoNN = convo_nn.episode_num
                    print(episode_number_ConvoNN)
                    start = time.time()
                    convo_nn.eval(envConvoNN, render)
                    end = time.time()
                    print("Convolutional Neural Network Time: " + str(end - start))
                    print("Convolutional Neural Network Reward: " + str(convo_nn.reward_sum))
                    fileconvoNN.write('%i,%.3f,%i\n' % (episode_number_NN, (end - start), convo_nn.reward_sum))
                    if episode_number_ConvoNN % record_rate_convoNN == 0:
                        print("Average ConvoNN time: " + str(sum(time_convonn) / len(time_convonn)))
                        print("Average ConvoNN reward: " + str(sum(reward_convonn) / len(reward_convonn)))
                        time_convonn, reward_convonn = [], []
                    else:
                        time_convonn.append(end - start)
                        reward_convonn.append(convo_nn.reward_sum)
                    if convo_nn.episode_num == 1000:
                        record_rate_convoNN = 1000
                        convo_nn.render_mod = 1000
                    elif convo_nn.episode_num == 10000:
                        record_rate_convoNN = 2000
                        convo_nn.render_mod = 2000

                if NeatAlgo:
                    # evaluate NEAT algorithm
                    episode_number_NEAT = neat.numEpisodes
                    print(episode_number_NEAT)
                    start = time.time()
                    neat.PongEvaluation()
                    end = time.time()
                    for genome in neat.population:
                        reward.append(genome.answer)
                    sortedPop = [x for _, x in sorted(zip(reward, neat.population), key=lambda pair: pair[0], reverse=True)]

                    print("Neat Algorithm Time: " + str(end - start))
                    print("Neat Algorithm Reward: " + str(sortedPop[0].answer))
                    time_neat.append(end-start)
                    reward_neat.append(sortedPop[0].answer)
                    if sortedPop[0].answer >= 0:
                        print('\n\n<<<<< I WON >>>>>\n\n')
                    if episode_number_NEAT % record_rate_NEAT == 0:
                        print("Average Neat Algorithm time: " + str(sum(time_neat) / len(time_neat)))
                        print("Average Neat Algorithm reward: " + str(sum(reward_neat) / len(reward_neat)))
                        if sortedPop:
                            ShowNEATNeuralNetwork(sortedPop[0], neat.nodeGeneArray, directory, neat.numEpisodes)
                        ShowSpeciesChart(neat.recordedSpecies, neat.numSpecies, directory, neat.numEpisodes)


        except KeyboardInterrupt:
            if neuralNet:
                fileNN.close()
            if convoNN:
                fileconvoNN.close()
            # if NeatAlgo:
                # print("...LOADING SPECIATION CHART...")
                # if sortedPop:
                #     ShowNEATNeuralNetwork(sortedPop[0], neat.nodeGeneArray, directory, neat.numEpisodes)
                # ShowSpeciesChart(neat.recordedSpecies, neat.numSpecies, directory, neat.numEpisodes)
            print('Interrupted')

if __name__ == '__main__':
    main()