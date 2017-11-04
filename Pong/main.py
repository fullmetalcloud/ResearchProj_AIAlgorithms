from neuralnet import NeuralNetwork, NNTest
from convonn import ConvolutionNN
from neatq import NEATPong, NeatTest
import tensorflow as tf
import gym
import time
import os
import itertools
import matplotlib.pyplot as pyplot

fig, ax = pyplot.subplots()

def main():
    # for testing
    render = False
    convoNN = False
    neuralNet = False
    NeatAlgo = False
    NNTests = False
    NEATTests = True

    render_mod = 10

    # hyperparameters
    numInputs = 6400
    numOutputs = 1
    num_layer_neurons_NN = [6400, 200, 1]
    num_layer_neurons_ConvoNN = [256, 200, 1]
    height = 80
    width = 80
    directory = os.getcwd() + "/tmp/"



    #parameters for monitoring and recording games
    episode_number_NN = 0
    time_nn, reward_nn  = [0], [0]
    episode_number_ConvoNN = 0
    time_convonn, reward_convonn = [0], [0]
    episode_number_NEAT = 0
    record_rate_NN = 100
    record_rate_convoNN = 100
    record_rate_NEAT = 100




    #initialize gym environment
    if neuralNet:
        sess_nn = tf.Session()
        envNN = gym.make('Pong-v0')
        envNN = gym.wrappers.Monitor(envNN, directory + "NN", force=True,
                                     video_callable=lambda episode_id: 0 == episode_number_NN % record_rate_NN)
        nn = NeuralNetwork(num_layer_neurons_NN, sess_nn)
    if convoNN:
        sess_convonn = tf.Session()
        envConvoNN = gym.make('Pong-v0')
        envConvoNN = gym.wrappers.Monitor(envConvoNN, directory + "ConvoNN", force=True,
                                          video_callable=lambda
                                              episode_id: 0 == episode_number_ConvoNN % record_rate_convoNN)
        convo_nn = ConvolutionNN(num_layer_neurons_ConvoNN, height, width, sess_convonn)
    if NeatAlgo:
        envNEAT = gym.make('Pong-v0')
        envNEAT = gym.wrappers.Monitor(envNEAT, directory+"NEAT", force=True,
                                 video_callable=lambda episode_id: 0 == episode_number_NEAT % record_rate_NEAT)
        neat = NEATPong(numInputs, numOutputs)

    # Unit testing of classes
    if NNTests:
        accuracy = 0
        numEpisodes = 0
        numTests = 10
        print("NNTests Beginning: ")
        for i in range(0, numTests):
            print("... testing " + str(i) + "...")
            sess_test = tf.Session()
            nnTest = NNTest(100, sess_test)
            acc, episodes = nnTest.test_eval()
            numEpisodes+=episodes
            accuracy+=acc
            sess_test.close()
            print(i, acc, episodes)
        print("Average NN Accuracy: " + str(accuracy/numTests))
        print("Average NN Episodes: " + str(numEpisodes/numTests))

    if NEATTests:
        arrGenome = "["
        nCount = 0
        numEpisodes = 0
        numTests = 10
        topGenomes = []
        print("NEATTests Beginning: ")
        for i in range(0,numTests):
            print("... testing " + str(i) + "...")
            neatTest = NeatTest()
            neatTest.XOREvaluation()

            population =neatTest.population

            print("results: " + str(i))
            print(len(neatTest.speciesGroups))
            reward = []
            for genome in population:
                reward.append(genome.answer)
            sortedPop = [x for _,x in sorted(zip(reward,population), key=lambda pair: pair[0], reverse=True)]
            genome = sortedPop[0]
            topGenomes.append(genome)
            for gene in genome.geneArray:
                arrGenome += str(gene.n1)
                arrGenome += ", "
                arrGenome += str(gene.n2)
                arrGenome += "; "
            print(arrGenome + "]")
            arrGenome = "["
            print(genome.weights)
            print(genome.activationList)
            print(genome.neuronArray)
            print(genome.answer)
            print(genome.tests)
            print()
            layers = {}
            neurons = []
            nodeGeneArray = neatTest.nodeGeneArray
            for v in set(nodeGeneArray.values()):
                layers[v] = []
            for k, v in nodeGeneArray.items():
                layers[v].append(k)
            largestLayerSize = max(len(x) for x in layers.values())
            for i, j in enumerate(genome.geneArray):
                neuron1 = j.n1
                neuron2 = j.n2
                neurons.append(neuron1)
                neurons.append(neuron2)
                layerN1 = layers[nodeGeneArray[neuron1]]
                layerN2 = layers[nodeGeneArray[neuron2]]
                n1 = layerN1.index(neuron1)
                n2 = layerN2.index(neuron2)
                type1 = neatTest.nodeGeneArray[neuron1]
                type2 = neatTest.nodeGeneArray[neuron2]
                posN1 = n1 * largestLayerSize/len(layerN1) + largestLayerSize/len(layerN1)/2
                posN2 = n2 * largestLayerSize/len(layerN2) + largestLayerSize/len(layerN2)/2
                line_y = (posN1, posN2)
                line_x = (type1, type2)
                if genome.activationList[i]:
                    if genome.weights[i] > 0:
                        pyplot.plot(line_x, line_y, linewidth=genome.weights[i], color='blue')
                    else:
                        pyplot.plot(line_x, line_y, linewidth=genome.weights[i], color='red')
                if type1 == 1:
                    pyplot.plot(type1, posN1, 'o', color='green')
                else:
                    pyplot.plot(type1, posN1, 'o', color='black')
                pyplot.plot(type2, posN2, 'o', color='black')

            pyplot.show()
            print(genome.answer)
            print(len(set(neurons)), neatTest.numEpisodes)
            nCount += len(set(neurons))
            numEpisodes += neatTest.numEpisodes
            print()
        print("Average Number of Nodes: " + str(nCount/numTests))
        print("Average Episodes: " + str(numEpisodes/numTests))

    else:
        while True:
            # Pong Game
            if neuralNet:
                episode_number_NN = nn.episode_num
                print(episode_number_NN)
                #evaluate Fully Connected Neural Network
                start = time.time()
                nn.eval(envNN, render)
                end = time.time()
                print("Neural Network Time: " + str(end - start))
                print("Neural Network Reward: " + str(nn.reward_sum))
                if episode_number_NN % record_rate_NN == 0:
                    print("Average NN time: " + str(sum(time_nn) / len(time_nn)))
                    print("Average NN reward: " + str(sum(reward_nn) / len(reward_nn)))
                    time_nn,reward_nn = [], []
                else:
                    time_nn.append(end - start)
                    reward_nn.append(nn.reward_sum)
                if episode_number_NN == 1000:
                    record_rate_NN = 1000
                elif episode_number_NN == 10000:
                    record_rate_NN = 2000

            if convoNN:
                #evaluate Convolutional Fully Connected Neural Network
                episode_number_ConvoNN = convo_nn.episode_num
                print(episode_number_ConvoNN)
                start = time.time()
                convo_nn.eval(envConvoNN, render)
                end = time.time()
                print("Convolutional Neural Network Time: " + str(end - start))
                print("Convolutional Neural Network Reward: " + str(convo_nn.reward_sum))
                if episode_number_ConvoNN % record_rate_convoNN == 0:
                    print("Average ConvoNN time: " + str(sum(time_convonn) / len(time_convonn)))
                    print("Average ConvoNN reward: " + str(sum(reward_convonn) / len(reward_convonn)))
                    time_convonn, reward_convonn = [], []
                else:
                    time_convonn.append(end - start)
                    reward_convonn.append(nn.reward_sum)
                if convo_nn.episode_num == 1000:
                    record_rate_convoNN = 1000
                elif convo_nn.episode_num == 10000:
                    record_rate_convoNN = 2000

            if NeatAlgo:
                y = 1
                y+=1
                y-=1

main()