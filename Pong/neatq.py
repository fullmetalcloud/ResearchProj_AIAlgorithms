
import numpy as np
import sys
import math
import operator
import random
import copy
from pong_image import *
import time
import gym
from multiprocessing import Process, Array, Manager, Pool
import tensorflow as tf

POP_SIZE = 20                             # Population size
STALEMATE_LIMIT = 20                        # Limit of Species Stalemate
SPECIES_THRESHOLD = 100                      # Initial Species delta if need more or less species
ADJUST_DELTA = 0.03                          # change in Species Delta (increase for more and vice versa)
GENE_SIZE_LIMIT = 6600                      # Gene size limit before gene size is taken into account
PERCENT_CHAMPS = 0.20                       # percent of genomes that are saved
SPECIES_SIZE = 10                            # Number of desired species before delta declination
NO_CHANGE_LIMIT = 20                       # Limit of generations a population does not change in fitness

ACTIVATION_PROB = 0.25                      # chance activation is second parent genome
NODE_PROB = 0.3                             # chance node mutation occurs to new genome
CONN_PROB = 0.7                             # chance connection mutation occurs to new genome
WEIGHT_PROB = 0.8                           # chance Weight mutation occurs to new genome
NEW_WEIGHT_PROB = 0.2                       # chance weight is changed completely
CROSS_MUTATION_PROB = 0.01                  # chance genome crosses over with another genome from another species

SIG_MOD = 1                                 # adjustment to sigmoid function
INIT_WEIGHT_VALUE = 5                       # largest starting weight of connection
WEIGHT_STDEV = 0.1                          # deviation and perturbation of weight
C1, C2, C3 = 0.5, 0.5, 0.3                      # constants of genome compatibility
INIT_FUNC = 'Relu'                          # initial function of layer evaluation (default is Sigmoid)
                                            # check list in layerEvaluation for names of functions
                                            # last layer is always evaluated with Sigmoid function

EXPECTED_ACCURACY = 0.9                     # (NEATTest only) expected accuracy of XOR test

# Neuron Gene shows unique id and what type of gene. 1 = input, 2 = output, 3 or more = hidden
# Node Gene array is an implicit dictionary of node neurons
# e.g. "neuron['1'] = 1" states neuron 1 is an input neuron

"""
/**** ConnectionGene ****/
By: Edward Guevara 
References:  Sources
Description: connection between two nodes and its innovation number (unique)
"""
class ConnectionGene():
    """Constructor for ConnectionGene"""
    def __init__(self,n1, n2, innovationNum, mutated=None):
        self.n1 = n1                            # incoming node
        self.n2 = n2                            # receiving node
        self.innovationNum = innovationNum      # innovation connection number (unique)
        self.mutated = mutated                  # innovation numbers that mutated this connection (done by node mutation)

"""
/**** Genome ****/
By: Edward Guevara 
References:  Sources
Description: array of connection genes with weights and activation list = neural network
"""
class Genome():
    """Constructor for Genome"""
    def __init__(self, geneArray, weights, activationList, genesMutated):
        self.geneArray = geneArray                      # array of connection genes,
        self.weights = weights                          # weights of connection genes
        self.activationList = activationList            # activation of connection gene
        self.genesMutated = genesMutated                # tracks the mutations that occurred in genome
        self.neuronArray = {}                           # for evaluation of NN
        self.fitness = 0.0                              # fitness value after evaluation
        self.answer = 0                                 # answer to problem
        self.test = 0                                   # for testing
        self.tests = []                                 # for testing

    """
    GenomeEval (BAD VERSION)
    brief: evaluates Genome NN
    input: none
    output: none
    """
    # def GenomeEval(self, inputArray, nodeGeneArray):
    #     layerArray = {}
    #     start = time.time()
    #     # setup neuron array to evaluate each node
    #     for connLoc, connGene in enumerate(self.geneArray):
    #         self.neuronArray[connGene.n1] = 0
    #         self.neuronArray[connGene.n2] = 0
    #         if not connGene.n2 in layerArray:
    #             layerArray[connGene.n2] = []
    #         layerArray[connGene.n2].append(connLoc)
    #     end = time.time()
    #     self.test += end - start
    #     print("SETUP TIME:" + str(end-start))
    #     # add input values to input neurons in neuron array
    #     for j, input in enumerate(inputArray):
    #         self.neuronArray[j] = input
    #     start = time.time()
    #     # evaluate each layer in order from smallest (inputs) to largest (outputs)
    #     test = sorted(nodeGeneArray.items(), key=lambda t: t[::-1])
    #     for nodeNum, layerNum in test:
    #         if nodeNum in layerArray:
    #             if layerNum > 1:
    #                 self.layerEvaluation(nodeNum, layerArray[nodeNum])
    #     end = time.time()
    #
    #     print("EVAL TIME: " + str(end - start))
    #     return
    #
    # """
    # layerEvaluation
    # brief: evaluates given layer of NN
    # input: nodeGeneArray
    # output:none
    # """
    # def layerEvaluation(self,nodeNum, connGenes):
    #     for j in connGenes:
    #         neuron1 = self.geneArray[j].n1
    #         if self.activationList[j]:
    #             self.neuronArray[nodeNum] += self.neuronArray[neuron1] * self.weights[j]
    #     value = self.neuronArray[nodeNum]
    #     try:
    #         self.neuronArray[nodeNum] = self.ActivationFunction(value)
    #     except:
    #         exit(1)
    #     return

    """
    GenomeEval (Working Version)
    brief: evaluates Genome NN
    input: none
    output: none
    """
    def GenomeEval(self, inputArray, nodeGeneArray):
        # setup neuron array to evaluate each node
        for node in nodeGeneArray.keys():
            self.neuronArray[node] = 0
        # add input values to input neurons in neuron array
        for j, input in enumerate(inputArray):
            self.neuronArray[j] = input
        # evaluate each layer in order from smallest (inputs) to largest (outputs)
        layerEval = self.layerEvaluation
        layers = sorted(set(nodeGeneArray.values()))
        for i in layers:
            if i == max(layers):
                layerEval(i, nodeGeneArray, 'Sigmoid')
            elif i > 1:
                layerEval(i, nodeGeneArray)
        return

    """
    layerEvaluation
    brief: evaluates given layer of NN
    input: nodeGeneArray
    output:none
    """
    def layerEvaluation(self, type, nodeGeneArray, func=INIT_FUNC):
        output = []
        # check each connection gene array
        for j, gene in enumerate(self.geneArray):
            # get node genes in the connection gene
            neuron1 = gene.n1
            neuron2 = gene.n2
            # try:
                # if gene is activated and receiving node is layer to evaluate, add sending node * weight
            if self.activationList[j] and nodeGeneArray[neuron2] == type:
                self.neuronArray[neuron2] += self.neuronArray[neuron1] * self.weights[j]
                output.append(neuron2)
            # except:
            #     print("layer_eval_collect",j, neuron2, len(self.activationList), len(nodeGeneArray))
            #     print([list(a.innovationNum for a in self.geneArray)])
            #     print(sys.exc_info()[0])
            #     exit(1)

        # Determine activation function
        if func == 'Relu':
            activationFunc = self.ReLUFunction
        elif func == 'ArcTan':
            activationFunc = self.ArcTanFunction
        else:
            activationFunc = self.SigmoidFunction

        # if not the input layer, evaluate nodes in layer with activation function (sigmoid func)
        for neuron in set(output):
            # try:
            value = self.neuronArray[neuron]
            self.neuronArray[neuron] = activationFunc(value)
            # except:
            #     print("layer_eval_activation_function", j, neuron, len(self.activationList), len(nodeGeneArray))
            #     print([list(a.innovationNum for a in self.geneArray)])
            #     print(sys.exc_info()[0])
            #     exit(1)
        return

    """
    GenomeEval (Tensorflow Version)
    brief: evaluates Genome NN using tensorflow
    input: inputArray, nodeGeneArray
    output:none 
    """
    # def GenomeEval(self,inputArray, nodeGeneArray):
    #     sparse = tf.SparseTensor(indices=self.geneArray, values=self.weights, shape=[len(nodeGeneArray), len(nodeGeneArray)])
    #     input =
    #     return

    """
    SigmoidFunction
    brief: activation function for neurons (hidden and output neurons)
    input: value
    output:answer to function 
    """
    def SigmoidFunction(self,value):
        return 1 / (1 + math.exp(-SIG_MOD * value))

    """
    ReLUFunction
    brief: activation function using ReLU equation
    input: value
    output:answer to function  
    """
    def ReLUFunction(self,value):
        return 0 if value < 0 else value

    """
    ArcTanFunction
    brief: activation function using inverse tangent
    input: value
    output:answer to equation
    """
    def ArcTanFunction(self, value):
        return math.atan(value)

"""
/**** Species ****/
By: Edward Guevara 
References:  Sources
Description: group of genomes that match in topology and weights
"""
class Species():
    """Constructor for Species"""
    def __init__(self,genome, speciesNum):
        self.niche = []                                             # genomes grouped with species
        self.topGenome = genome                                     # best genome of the current and previous species
        self.speciesNum = speciesNum                                # species number
        self.niche.append(genome)
        self.stalemate = 1                                          # number of times species does not improve
        self.adjustedFitness = 0                                    # adjusted fitness based on niche and fitness
        self.prevAdjustedFitness = 0                                # previous fitness to check for stalemate
        self.numOffspring = 0                                       # number of offspring made

    """
    Reset
    brief: deletes all but champions of Species
    input: 
    output:none 
    """
    def Reset(self):
        # check if species stalemate from previous generation
        if abs(self.adjustedFitness - self.prevAdjustedFitness) < 0.001:
            self.stalemate += 1
        else:
            self.stalemate = 1
        # reset niche and put top genome in niche
        if self.niche:
            self.topGenome = self.niche[0]
        self.niche = []
        self.niche.append(self.topGenome)
        return

    """
    AdjustedFitness
    brief: adjusts fitness function based on Species population and scores
    input: 
    output:none 
    """
    def AdjustedFitness(self):
        self.adjustedFitness = 0
        # check if stalemate, punish if stalemate limit is reached
        if self.stalemate >= STALEMATE_LIMIT:
            stalemateAdjustment = np.square(self.stalemate)
        else:
            stalemateAdjustment = self.stalemate
        # reevaluate fitness of all genomes in niche
        if self.niche:
            for genome in self.niche:
                genome.fitness = genome.fitness/len(self.niche)
                self.adjustedFitness+= genome.fitness/stalemateAdjustment
        return

    """
    GenomeCompatibility
    brief: checks if genome fits within Species based on first Species
    input: newGenome
    output:True or False 
    """
    def GenomeCompatibility(self,newGenome, speciesThreshold):
        disjoint, excess, totalDiffWeights = 0, 0, 0
        c1, c2, c3 = C1, C2, C3

        # get gene arrays to compare (topGenome and genome to compare)
        topGeneArray = self.topGenome.geneArray
        newGenomeArray = newGenome.geneArray
        topCount, newCount = 0, 0

        # add disjoint if two genes' innovation number are not equal or compare weights of genes
        while topCount < len(topGeneArray) and newCount < len(newGenomeArray):
            if topGeneArray[topCount].innovationNum > newGenomeArray[newCount].innovationNum:
                newCount+=1
                disjoint+=1
            elif topGeneArray[topCount].innovationNum < newGenomeArray[newCount].innovationNum:
                topCount += 1
                disjoint += 1
            else:
                totalDiffWeights += abs(self.topGenome.weights[topCount] - newGenome.weights[newCount])
                topCount+=1
                newCount+=1
        excess = abs(len(newGenomeArray)- len(topGeneArray))

        # gene size is the biggest genome
        totalGenes = len(newGenomeArray) if (len(newGenomeArray) > len(topGeneArray)) else len(topGeneArray)
        if totalGenes < GENE_SIZE_LIMIT:
            totalGenes = 1

        # evaluation equation
        differenceVal = c1 * excess / totalGenes + c2 * disjoint / totalGenes + c3 * totalDiffWeights / totalGenes

        if speciesThreshold > differenceVal:
            return True
        else:
            return False

"""
/**** Population ****/
By: Edward Guevara 
References:  Sources
Description: Population of Genomes, used for Species Evaluation
"""
class Population():
    """Constructor for Population"""
    def __init__(self, numInputs, numOutputs):
        self.population, self.connGeneArray = [], []
        self.nodeGeneArray, self.speciesGroups = {}, {}

        # initialize node gene array and connection gene array
        innovationNum = 0
        for i in range(0, numInputs):
            self.nodeGeneArray[i] = 1
        for j in range(numInputs, numInputs + numOutputs):
            self.nodeGeneArray[j] = 2
            for k in range(0, numInputs):
                connGene = ConnectionGene(k, j, innovationNum)
                self.connGeneArray.append(connGene)
                innovationNum += 1

        self.ResetPopulation()
        self.numSpecies = 0
        self.deltaSpecies = SPECIES_THRESHOLD
        self.total = 0
        self.noChange =0
        self.numEpisodes = 0

    """
    ResetPopulation
    brief: resets population with minimal topologies
    input: numInputs, numOutputs
    output:none 
    """
    def ResetPopulation(self):
        self.population = []
        weights, activationList, genesMutated = [], [], []
        for q in range(0, POP_SIZE):
            for w in self.connGeneArray:
                weights.append(self.InitWeight())
                activationList.append(True)
                genesMutated.append(False)
            geneArray = self.connGeneArray[:]
            self.population.append(Genome(geneArray, weights, activationList, genesMutated))
            weights, activationList, genesMutated = [], [], []
        return

    """
    AddSpecies
    brief: Adds a new species to population
    input: newSpecies
    output:void 
    """
    def AddSpecies(self,newGenomeSpecies):
        newSpecies = Species(newGenomeSpecies,self.numSpecies)
        self.speciesGroups[self.numSpecies] = newSpecies
        self.numSpecies += 1
        return

    """
    WeightedChoice
    brief: Randomly chooses element based on weight 
    (based on https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice)
    input: choice
    output: i 
    """
    def WeightedChoice(self,choices):
        total = sum(w for c, w in choices)
        r = random.uniform(0.0, total)
        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto+=w
        assert False, "Shouldn't get here"

    """
    BreedNewOffspring
    brief: makes new offspring through mating and/or mutation
    input: species
    output: newGenome 
    """
    def BreedNewOffspring(self,species):
        newGenome = None
        if 1 < len(species.niche):
            g1, g2 = 0, 0
            newW, newArr, newActivationList, newGenesMutated = [], [], [], []

            genomeFitnesses = list(g.fitness for g in species.niche)
            genome1 = self.WeightedChoice(list(zip(species.niche, genomeFitnesses)))
            # cross species mutation
            if random.uniform(0, 1) <= CROSS_MUTATION_PROB:
                speciesX = random.choice(self.speciesGroups)
                while not speciesX.niche:
                    speciesX = random.choice(self.speciesGroups)
                genomeFitnessesX = list(g.fitness for g in speciesX.niche)
                genome2 = self.WeightedChoice(list(zip(speciesX.niche, genomeFitnessesX)))
            else:
                genome2 = self.WeightedChoice(list(zip(species.niche, genomeFitnesses)))
            gArr1 = genome1.geneArray
            gArr2 = genome2.geneArray
            while (g1 < len(gArr1) and g2 < len(gArr2)):
                # if genome1 is ahead of genome2, add genome2.gene and increment next place in genome2
                if gArr1[g1].innovationNum > gArr2[g2].innovationNum:
                    newArr.append(gArr2[g2])
                    newW.append(genome2.weights[g2])
                    newActivationList.append(genome2.activationList[g2])
                    newGenesMutated.append(genome2.genesMutated[g2])
                    g2 += 1
                # if genome1 is behind of genome2, add genome1.gene and increment next place in genome1
                elif gArr1[g1].innovationNum < gArr2[g2].innovationNum:
                    newArr.append(gArr1[g1])
                    newW.append(genome1.weights[g1])
                    newActivationList.append(genome1.activationList[g1])
                    newGenesMutated.append(genome1.genesMutated[g1])
                    g1 += 1
                # have the same gene in both genomes, add and check if both have gene activated
                # if not, deactivate it or set false in activation list
                else:
                    newArr.append(gArr1[g1])
                    if np.random.uniform(0, 1) > 0.5:
                        newW.append(genome1.weights[g1])
                    else:
                        newW.append(genome2.weights[g2])

                    if (genome1.activationList[g1] and genome2.activationList[g2]):
                        # and np.random.randint(0, int(1 / ACTIVATION_PROB)) == 1:
                        newActivationList.append(True)
                    else:
                        newActivationList.append(False)
                    if (not genome1.genesMutated[g1] and not genome2.genesMutated[g2]):
                        newGenesMutated.append(False)
                    else:
                        newGenesMutated.append(True)
                    g1 += 1
                    g2 += 1
            # add any excess genes from either genomes
            while g1 < len(gArr1):
                newArr.append(gArr1[g1])
                newW.append(genome1.weights[g1])
                newActivationList.append(genome1.activationList[g1])
                newGenesMutated.append(genome1.genesMutated[g1])
                g1 += 1
            while g2 < len(gArr2):
                newArr.append(gArr2[g2])
                newW.append(genome2.weights[g2])
                newActivationList.append(genome2.activationList[g2])
                newGenesMutated.append(genome2.genesMutated[g2])
                g2 += 1
            # create new genome
            newGenome = Genome(newArr, newW, newActivationList, newGenesMutated)
        elif 1 == len(species.niche):
            newGenome = species.niche[0]
            self.WeightMutation(newGenome)
        if random.uniform(0, 1) <= NODE_PROB:
            self.NodeMutation(newGenome)
        if random.uniform(0, 1) <= CONN_PROB:
            self.ConnectionMutation(newGenome)
        if random.uniform(0, 1) <= WEIGHT_PROB:
            self.WeightMutation(newGenome)
        newGenome.geneArray.sort(key=lambda connGene: connGene.innovationNum)
        return newGenome

    """
    Crossover
    brief: Crossbreeds genomes within same Species
    input:
    output:void
    """
    def Crossover(self):
        newPop =  []
        breed = self.BreedNewOffspring
        for species in self.speciesGroups.values():
            if species.niche:
                count = 0
                while count < species.numOffspring and POP_SIZE > len(newPop):
                    newGenome = breed(species)
                    if newGenome != None:
                        newPop.append(newGenome)
                    count += 1
        self.population = newPop
        return

    """
    CheckThreshold
    brief: adjusts threshold for number of species
    input: 
    output:none 
    """
    def CheckThreshold(self,):
        if SPECIES_SIZE < len(self.speciesGroups):
            self.deltaSpecies += ADJUST_DELTA
        else:
            self.deltaSpecies -= ADJUST_DELTA
        if self.deltaSpecies < ADJUST_DELTA:
            self.deltaSpecies = ADJUST_DELTA
        return

    """
    Speciation
    brief: organize population into species and creates new population
    input: 
    output:none 
    """
    def Speciation(self):
        oldTotal = self.total
        self.total = 0
        # check all genomes
        for species in self.speciesGroups.values():
            # check = self.KillSpecies(species)
            species.Reset()
        self.CheckThreshold()
        for genome in self.population:
            # add new Species if species list is empty
            if not self.speciesGroups:
                self.speciesGroups[self.numSpecies] = Species(genome, self.numSpecies)
                self.numSpecies += 1
            else:
                check = False
                count = min(self.speciesGroups.keys())
                while (count < len(self.speciesGroups)) and not check:
                    species = self.speciesGroups[count]
                    check = species.GenomeCompatibility(genome, self.deltaSpecies)
                    count+=1
                if not check:
                    self.AddSpecies(genome)
                else:
                    species.niche.append(genome)
                    check = False

        for species in self.speciesGroups.values():
            species.AdjustedFitness()
            self.total+=species.adjustedFitness

        if abs(self.total-oldTotal) < 0.01:
            self.noChange+=1

        if self.noChange > NO_CHANGE_LIMIT:
            self.noChange = 0
            topTwo = sorted(self.speciesGroups.values(),
                                 key=lambda s: s.adjustedFitness, reverse=True)
            for species in self.speciesGroups.values():
                # determine number of offspring
                species.numOffspring = 0
                # keep top 20% in niche
                size = len(species.niche) * 0.2
                if size > 2:
                    species.niche = species.niche[:int(size) - 1]
            for s in topTwo[:2]:
                # determine number of offspring
                s.numOffspring = POP_SIZE / 2
                # sort niche for elimination
                s.niche.sort(key=lambda genome: genome.fitness, reverse=True)

        else:
            for species in self.speciesGroups.values():
                # determine number of offspring
                species.numOffspring = species.adjustedFitness/self.total*POP_SIZE
                # sort niche for elimination
                species.niche.sort(key=lambda genome: genome.fitness, reverse=True)

                # keep top 20% in niche
                size = len(species.niche) * PERCENT_CHAMPS
                if size > 2:
                    species.niche = species.niche[:int(size) - 1]

        return

    """
    InitWeight
    brief: initializes a new weight
    input: 
    output:value 
    """
    def InitWeight(self):
        value = random.uniform(-INIT_WEIGHT_VALUE, INIT_WEIGHT_VALUE)
        return value

    """
    NodeMutation
    brief: Mutates a genome with a new node
    input: genome
    output:none 
    """
    def NodeMutation(self,genome):
        activated = True
        array = genome.geneArray
        chosenGene = 0

        # find activated connection gene and deactivate it
        while activated and chosenGene < (len(array)):
            activated = genome.genesMutated[chosenGene]
            if activated:
                chosenGene += 1

        # if all connection genes have been mutated in array
        if chosenGene == len(array):
            return
        genome.genesMutated[chosenGene] = True

        # if mutated, add connection genes that mutated the connection gene to the genome
        if self.connGeneArray[array[chosenGene].innovationNum].mutated:
            n1 = self.connGeneArray[array[chosenGene].innovationNum].mutated[0]
            n2 = self.connGeneArray[array[chosenGene].innovationNum].mutated[1]
            genome.geneArray.append(self.connGeneArray[n1])
            genome.weights.append(1)
            genome.activationList.append(True)
            genome.genesMutated.append(False)

            genome.geneArray.append(self.connGeneArray[n2])
            genome.weights.append(genome.weights[chosenGene])
            genome.activationList.append(True)
            genome.genesMutated.append(False)
        else:

            # get node genes and check if
            n1 = array[chosenGene].n1
            n2 = array[chosenGene].n2

            # update layers if adding new layer
            if(self.nodeGeneArray[n2] - self.nodeGeneArray[n1] == 1):
                for n, layerVal in self.nodeGeneArray.items():
                    if layerVal >= self.nodeGeneArray[n2]:
                        self.nodeGeneArray[n] += 1

            # make new node
            innovationNum = len(self.connGeneArray)
            newNode = max(self.nodeGeneArray.keys()) + 1
            self.nodeGeneArray[newNode] = self.nodeGeneArray[n2] - 1

            # add 2 new connection genes to connection gene array and to genome
            connGene1 = ConnectionGene(n1, newNode,innovationNum)
            self.connGeneArray.append(connGene1)
            genome.geneArray.append(connGene1)

            genome.weights.append(1)
            genome.activationList.append(True)
            genome.genesMutated.append(False)

            innovationNum += 1
            connGene2 = ConnectionGene(newNode, n2, innovationNum)
            self.connGeneArray.append(connGene2)
            genome.geneArray.append(connGene2)
            genome.weights.append(genome.weights[chosenGene])
            genome.activationList.append(True)
            genome.genesMutated.append(False)

            self.connGeneArray[array[chosenGene].innovationNum].mutated = (innovationNum - 1, innovationNum)
            array[chosenGene].mutated = (innovationNum - 1, innovationNum)
        return

    """
    ConnectionMutation
    brief: Mutates a genome to add a new connection
    input: genome
    output:none 
    """
    def ConnectionMutation(self,genome):
        n1 = 0
        n2 = 0
        choice = False
        count = 0
        check = False
        array = genome.geneArray
        newConnGene = None
        while not choice and count < len(array):
            n1 = random.choice(array).n1
            n2 = random.choice(array).n2
            if n1 == n2:
                choice = False
            else:
                for gene in array:
                    if gene.n1 == n1 and gene.n2 == n2 and genome.activationList[count] :
                        if (self.nodeGeneArray[n2] - self.nodeGeneArray[n1]) == 1:
                            choice = False
                            count +=1
                            break
                        else:
                            genome.activationList[count] = False
                            return
                    else:
                        choice = True
        if count == len(array):
            return
        for connGene in self.connGeneArray:
            if connGene.n1 == n1 and connGene.n2 == n2:
                if not genome.activationList[count]:
                    genome.activationList[count] = True
                    check = True
                    break
                else:
                    newConnGene = connGene
                    innoNum = newConnGene.innovationNum
                    genome.geneArray.insert(innoNum, newConnGene)
                    genome.activationList.insert(innoNum, True)
                    genome.weights.insert(innoNum, self.InitWeight())
                    genome.genesMutated.insert(innoNum, False)
                    check = True
                    break
        if not check:
            newConnGene = ConnectionGene(n1, n2, self.connGeneArray[-1].innovationNum+1)
            self.connGeneArray.append(newConnGene)
            try:
                if newConnGene == None:
                    raise Exception("conn mutation")
                genome.geneArray.append(newConnGene)
                genome.weights.append(self.InitWeight())
                genome.activationList.append(True)
                genome.genesMutated.append(False)
            except:
                print("conn mutation error")
                exit(1)
        return

    """
    WeightMutation
    brief: mutates the weight of a genome
    input: genome
    output:none 
    """
    def WeightMutation(self,genome):
        if random.uniform(0, 1) <= NEW_WEIGHT_PROB:
            genome.weights[np.random.randint(0, len(genome.weights))] = self.InitWeight()
        else:
            genome.weights[np.random.randint(0, len(genome.weights))]\
                += random.uniform(-INIT_WEIGHT_VALUE * WEIGHT_STDEV, INIT_WEIGHT_VALUE * WEIGHT_STDEV)
        return

"""
/**** NEATPong ****/
By: Edward Guevara 
References:  Sources
Description: NEAT Algorithm Implementation using Pong
"""
class NEATPong(Population):
    """Constructor for NEAT"""
    def __init__(self, numInputs, numOutputs, manager, env, render=False, render_mod=100):
        self.env = env
        self.numInputs = numInputs
        self.render = render
        self.render_mod = render_mod
        self.manager = manager
        self.sharedList = manager.list(range(POP_SIZE))
        super().__init__(numInputs, numOutputs)

    """
    chooseAction
    brief: picks Action for Pong
    input: probability
    output: action (2 = up, 3 = down) 
    """
    def chooseAction(self,probability):
        randomValue = np.random.uniform()
        if randomValue < probability:
            # signifies up in openai gym
            return 2
        else:
            # signifies down in openai gym
            return 3

    def discount_rewards(self,rewards, gamma):
        """ Actions you took 20 steps before the end result are less important to the overall result than an action 
        you took a step ago. This implements that logic by discounting the reward on previous actions based on how 
        long ago they were taken"""
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def discount_with_rewards(self,gradient_log_p, episode_rewards, gamma):
        """ discount the gradient with the normalized rewards """
        discounted_episode_rewards = self.discount_rewards(episode_rewards, gamma)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return gradient_log_p * discounted_episode_rewards

    """
    PongEvaluation
    brief: evaluates generation of NEAT
    input: 
    output: 
    """
    def PongEvaluation(self):
        processes = []
        population = self.sharedList
        fitnessFunction = self.PongFitnessFunction
        speciation = self.Speciation
        crossover = self.Crossover
        popSort = self.population.sort
        start = time.time()
        # process: SPECIATION -> CROSSOVER -> EVALUATION -> CHECK -> SPECIATION...
        for i, genome in enumerate(self.population):
            if i != 0:
                env = gym.make('Pong-v0')
                p = Process(target=fitnessFunction, args=(population, self.nodeGeneArray, genome, i, env))
                processes.append(p)
        for p in processes:
            p.start()
        fitnessFunction(population, self.nodeGeneArray, self.population[0], 0, self.env)
        for p in processes:
            p.join()
        print("PARALLEL TIME: %.12f" % (time.time() - start))
        self.population = list(population[:])
        # start = time.time()
        # for i, genome in enumerate(self.population):
        #     fitnessFunction(population, self.nodeGeneArray, genome, i, self.env)
        # print("SEQUENTIAL TIME: %.12f" % (time.time()-start))
        self.numEpisodes += 1
        start = time.time()
        speciation()
        print("SPECIATION TIME: %.12f" % (time.time()-start))
        start = time.time()
        crossover()
        print("CROSSOVER TIME: %.12f" % (time.time() - start))
        popSort(key=lambda g: g.answer, reverse=True)
        # show information needed to be seen
        print(self.numEpisodes , len(self.speciesGroups))
        print(len(self.nodeGeneArray), len(self.connGeneArray))
        arrGenome = "["
        g = self.population[0]
        print(g.answer)
        # print(g.tests)
        for gene in g.geneArray:
            arrGenome += str(gene.n1)
            arrGenome += ", "
            arrGenome += str(gene.n2)
            arrGenome += "; "
        print(arrGenome + "]")
        return

    """
    PongFitnessFunction
    brief: evaluates population for Pong
    input: env, inputDimensions, nodeGeneArray, generation, genome, render, render_mod
    output:none 
    """

    def PongFitnessFunction(self, population, nodeGeneArray, genome, i, env):
        # print("Genome running: %i" % i)
        start = time.time()
        observation = env.reset()
        rewardSum = 1
        done = False
        prev_processed_observations = None
        eval = genome.GenomeEval
        maxNode = max(nodeGeneArray, key=nodeGeneArray.get)
        process = preprocess_observations
        render = i == 0 and self.render and self.numEpisodes % self.render_mod == 0
        while not done:
            if render:
                env.render()
            processed_observations, prev_processed_observations = process(observation,
                                                                          prev_processed_observations,
                                                                          self.numInputs)
            eval(processed_observations, nodeGeneArray)
            up_probability = genome.neuronArray[maxNode]
            action = self.chooseAction(up_probability)
            # carry out the chosen action
            observation, reward, done, info = env.step(action)
            rewardSum += reward

            # see here: http://cs231n.github.io/neural-networks-2/#losses
            fake_label = 1 if action == 2 else 0
            loss_function_gradient = fake_label - up_probability

        # TODO: determine fitness function and Apply Q-Learning
        genome.fitness = np.square((21.0 + rewardSum)/43.0)
        genome.answer = rewardSum
        env.close()
        end = time.time()
        population[i] = genome
        # print("GENOME %i Time: %.12f" % (i, end-start))
        # print("GENOME %i Fitness: %.12f" % (i, population[i].answer))
        return

"""
/**** NeatTest ****/
By: Edward Guevara 
References:  Sources
Description: tests the implementation of the NEAT algo using XOR example
"""
class NeatTest(Population):
    """Constructor for NeatTest"""
    def __init__(self):
        numInputs = 3
        numOutputs = 1
        super().__init__(numInputs, numOutputs)

    """
    XORFitnessFunction
    brief: evaluates population for XOR example
    input: genome
    output:none 
    """

    def XORFitnessFunction(self,nodeGeneArray, genome):
        total = 0
        fitness = 0
        genome.tests = []
        input = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        output = [0, 1, 1, 0]
        for model in range(0, 4):
            genome.GenomeEval(input[model], nodeGeneArray)
            ans = genome.neuronArray[max(nodeGeneArray, key = nodeGeneArray.get)]
            total += 1 - abs(output[model] - ans)
            fitness += 1 - np.square(output[model] - ans)
            genome.tests.append(ans)
        genome.answer = total / 4
        genome.fitness = fitness
        genome.test = genome.test/4
        return

    """
    XOREvaluation
    brief: evaluation of XOR example
    input: 
    output:none 
    """
    def XOREvaluation(self):
        speciation = self.Speciation
        crossover = self.Crossover
        sortPopulation = self.population.sort
        fitnessFunction = self.XORFitnessFunction
        check = False
        test = 0
        arrGenome = ""
        for j, genome in enumerate(self.population):
            fitnessFunction(self.nodeGeneArray, genome)
        while not check:
            # if self.numEpisodes % 500 == 1:
            #     self.ResetPopulation()
            #     for genome in self.population:
            #         self.XORFitnessFunction(self.nodeGeneArray, genome)

            # show information needed to be seen
            # print(self.numEpisodes , len(self.speciesGroups))
            # g = self.population[0]
            # print(g.answer)
            # print(g.tests)
            # for gene in g.geneArray:
            #     arrGenome += str(gene.n1)
            #     arrGenome += ", "
            #     arrGenome += str(gene.n2)
            #     arrGenome += "; "
            # print(arrGenome + "]")
            # arrGenome = "["
            self.numEpisodes +=1

            # process: SPECIATION -> CROSSOVER -> EVALUATION -> CHECK -> SPECIATION...
            start = time.time()
            speciation()
            # end = time.time()
            # print("SPECIATION TIME: %.12f" % (end-start))
            # start = time.time()
            crossover()
            # end = time.time()
            # print("CROSSOVER TIME: %.12f" % (end-start))
            start = time.time()
            for genome in self.population:
                fitnessFunction(self.nodeGeneArray, genome)
            end = time.time()
            print("EVALUATION TIME: %.12f" % (end-start))
            sortPopulation(key=lambda g: g.answer, reverse=True)
            for genome in self.population:
                if genome.answer > EXPECTED_ACCURACY:
                    check = True
                    break
        return
