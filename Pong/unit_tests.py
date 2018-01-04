import time
import os

import tensorflow as tf
from Agents.fully_conn_nn import NNAgent
from Agents.poli_grad_nn import PoliGradAgent
from NEAT.neatq import *
from NEAT.plot import *

# location of saved files (networks, speciation graphs, videos)
directory = os.getcwd() + "/tmp/"

"""
UnitTest_NNAgent
brief: tests NNAgent using XOR example
input:
output:none 
"""
def UnitTest_NNAgent():
    accuracy = 0
    numEpisodes = 0
    numTests = 10
    layers = [2, 2, 1]
    print("NNTests Beginning: ")
    for i in range(0, numTests):
        print("... testing " + str(i) + "...")
        sess_test = tf.Session()
        nnTest = NNAgent(layers, sess_test)
        acc, episodes = nnTest.test_eval()
        numEpisodes += episodes
        accuracy += acc
        sess_test.close()
        print(i, acc, episodes)
    print("Average NN Accuracy: " + str(accuracy / numTests))
    print("Average NN Episodes: " + str(numEpisodes / numTests))
    return

"""
UnitTest_NEAT
brief: tests NEAT Agent using XOR example
input: 
output: 
"""
def UnitTest_NEAT():
    arrGenome = "["
    nCount = 0
    numEpisodes = 0
    avgTime = 0
    numTests = 1000
    topGenomes = []
    print("NEATTests Beginning: ")
    for i in range(0, numTests):
        manager = Manager()
        print("... testing " + str(i) + "...")
        neatTest = NeatTest(manager)
        start = time.time()
        neatTest.XOREvaluation()
        end = time.time()
        population = neatTest.population
        print("results: " + str(i))
        print()
        reward = []
        for genome in population:
            reward.append(genome.answer)
        sortedPop = [x for _, x in sorted(zip(reward, population), key=lambda pair: pair[0], reverse=True)]
        genome = sortedPop[0]
        topGenomes.append(genome)
        # print(len(neatTest.speciesGroups))
        # for gene in genome.geneArray:
        #     arrGenome += str(gene.n1)
        #     arrGenome += ", "
        #     arrGenome += str(gene.n2)
        #     arrGenome += "; "
        # print(arrGenome + "]")
        # arrGenome = "["
        # print(genome.weights)
        # print(genome.activationList)
        # print(genome.neuronArray)
        # print(genome.answer)
        # print(genome.tests)
        neurons = []
        ShowNEATNeuralNetwork(genome, neatTest.nodeGeneArray, directory, i)
        ShowSpeciesChart(neatTest.recordedSpecies, neatTest.numSpecies, directory, i)
        for i, j in enumerate(genome.geneArray):
            neurons.append(j.n1)
            neurons.append(j.n2)
        print("ACCURACY: % .6f" % genome.answer)
        print("SPECIES NUMBER: %i" % genome.speciesNum)
        print("GENOME HIDDEN NEURONS: %i" % (len(set(neurons)) - 4))
        print("NUMBER OF EPISODES: %i" % neatTest.numEpisodes)
        print("NUMBER OF SPECIES: %i" % neatTest.numSpecies)
        print(str(end - start))
        nCount += len(set(neurons))
        numEpisodes += neatTest.numEpisodes
        avgTime += (end - start)
        print()
    print("Average Number of Nodes: " + str(nCount / numTests))
    print("Average Episodes: " + str(numEpisodes / numTests))
    print("Average Time: " + str(avgTime / numTests))
    return

"""
UnitTest_PoliGradNN
brief: test the PoliGradNN using pole balancing test
input: 
output: 
"""
def UnitTest_PoliGradNN():
    accuracy = 0
    numEpisodes = 0
    layers = [4, 10, 1]
    print("NNTests Beginning: ")
    print("... testing PoliGradNN ...")
    env = gym.make('CartPole-v0')
    sess_test = tf.Session()
    nnTest = PoliGradAgent(layers, sess_test)
    acc, episodes = nnTest.test_eval(env)
    numEpisodes += episodes
    accuracy += acc
    sess_test.close()
    print(acc, episodes)
    print("Average NN Accuracy: " + str(accuracy))
    print("Average NN Episodes: " + str(numEpisodes))
    return