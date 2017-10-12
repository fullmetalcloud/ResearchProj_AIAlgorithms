
import numpy as np
import sys
import math

CONNECTIONPROB = 0.03
NODEPROB = 0.005
ACTIVATIONPROB = 0.05
WEIGHTPROB = 0.05
POPSIZE = 100
NUMCHAMPS = 10
RESETGENOMES = 10
NUMINPUTS = 2
NUMOUTPUTS = 1
INTERATION = 1000
EXCESSCOEFFICIENT = 1
DISJOINTCOEFFICIENT = 1
AVGWEIGHTCOEFFICIENT = 0.4
SPECIESTHRESHOLD = 3.0

# Neuron Gene shows what type of gene. 1 = input, 2 = hidden, 3 = output

class ConnectionGene:
    def __init__(self, n1, n2, innovationNum):
        self.n1 = n1
        self.n2 = n2
        self.innovationNum = innovationNum

# Genome holds a list of Genes. The neuron array is to hold values evaluated by the Genome
# geneArray holds connections between nodes. associated with connectionGene
class Genome:
    def __init__(self, geneArray, weights, activationList):
        self.geneArray = geneArray
        self.weights = weights
        self.activationList = activationList
        self.neuronArray = []
        for i in range(0,1000):
            self.neuronArray.append(0)

    def LayerEvaluation(self, type, nodeGeneArray):
        for j in range(0,len(self.geneArray)):
            neuron1 = self.geneArray[j].n1
            neuron2 = self.geneArray[j].n2
            try:
                if self.activationList[j] and nodeGeneArray[neuron2] == type:
                    self.neuronArray[neuron2] += self.neuronArray[neuron1]*self.weights[j]
            except:
                print("layer_eval",j, neuron2, len(self.activationList), len(nodeGeneArray))
                print([list(a.innovationNum for a in self.geneArray)])
                print(sys.exc_info()[0])
                exit(1)
        if type == 2 or type == 3:
            for j in range(0, len(self.geneArray)):
                neuron1 = self.geneArray[j].n1
                neuron2 = self.geneArray[j].n2
                try:
                    if self.activationList[j] and nodeGeneArray[neuron2] == type:
                        value = self.neuronArray[neuron2]
                        self.neuronArray[neuron2] = 1 / (1 + math.exp(-value))
                except:
                    print("layer_eval", j, neuron2, len(self.activationList), len(nodeGeneArray))
                    print([list(a.innovationNum for a in self.geneArray)])
                    print(sys.exc_info()[0])
                    exit(1)
    #fitness/reward function for genetic crossover
    def FitnessEvaluation(self, nodeGeneArray):


#group of genomes that match in topology and weights
#created at beginning of Crossover and when Species goes over SPECIESTHRESHOLD with other populations
class Species:
    def __init__(self, genome):
        self.niche = []
        self.niche.append(genome)

    def AdjustedFitness(self,genome):

        return
    def GenomeCompatibility(self,comparedGenome):


        if SPECIESTHRESHOLD > differenceVal:
            return True
        else:
            return False


#Population of Genomes
class Population:
    def __init__(self, species):
        self.topSpecies = species
        self.numSpecies = 1
        self.population = []
        self.population.append(species)



class NEAT:
    def __init__(self):

    def Eval(self):
        values = []
        for i, genome in enumerate(self.populationArray):
            value = [i, genome.fitnesseval(inputValues, self.nodeGeneArray, actual)]
            values.append(value)
        self.evaluationValues = values
        print(values[0][1])
        return values
        return

    def Crossover(self):
        return
