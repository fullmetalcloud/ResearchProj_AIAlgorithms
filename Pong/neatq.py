
import numpy as np
import sys
import math

SPECIESTHRESHOLD = 10


# Neuron Gene shows what type of gene. 1 = input, 2 = hidden, 3 = output

"""
/**** ConnectionGene ****/
By: Edward Guevara 
References:  Sources
Description: connection between two nodes and its innovation number (unique)
"""
class ConnectionGene():
    """Constructor for ConnectionGene"""
    def __init__(self,n1, n2, innovationNum):
        self.n1 = n1
        self.n2 = n2
        self.innovationNum = innovationNum


"""
/**** Genome ****/
By: Edward Guevara 
References:  Sources
Description: array of connection genes with weights and activation list = neural network
"""
class Genome():
    """Constructor for Genome"""
    def __init__(self,geneArray, weights, activationList):

        #TODO USE MATRIX EVALUATION FOR

    """
    GenomeEval
    brief: evaluates Genome NN
    input: none
    output: none 
    """
    def GenomeEval(self):

        return

    """
    FitnessEval
    determines the fitness value of the genome
    input: 
    output:none 
    """
    def FitnessEval(self):

        return

"""
/**** Species ****/
By: Edward Guevara 
References:  Sources
Description: group of genomes that match in topology and weights
"""
class Species():
    """Constructor for Species"""
    def __init__(self,genome):
        self.niche = []
        self.niche.append(genome)

    """
    AdjustedFitness
    brief: adjusts fitness function based on Species population and scores
    input: 
    output:none 
    """
    def AdjustedFitness(self,):

        return
    """
    GenomeCompatibility
    brief: checks if genome fits within Species based on first Species
    input: comparedGenome
    output:True or False 
    """
    def GenomeCompatibility(self,comparedGenome):
        differenceVal = 0

        if SPECIESTHRESHOLD > differenceVal:
            return True
        else:
            return False

"""
/**** Population ****/
By: Edward Guevara 
References:  Sources
Description: Population of Genomes
"""
class Population():
    """Constructor for Population"""
    def __init__(self,species):
        self.numSpecies = 1
        self.population = []
        self.population.append(species)

    """
    AddSpecies
    brief: Adds a new species to population
    input: newSpecies
    output:void 
    """
    def AddSpecies(self,newSpecies):

        return

    """
    KillSpecies
    brief: if species has less than given Minimum for given generations, Species ends and genomes die
    input: species
    output:
    """
    def KillSpecies(self,species):

        return

    """
    Crossover
    brief: Crossbreeds genomes within same Species
    input: newPopulation
    output:void 
    """
    def Crossover(self,newPopulation):

        return

    """
    CrossMutation
    brief: Possible Crossbreed between Species
    input: species1, species2
    output:genome 
    """
    def CrossMutation(self,species1, species2):

        return

"""
/**** NEAT ****/
By: Edward Guevara 
References:  Sources
Description: NEAT Algorithm Implementation
"""
class NEAT():
    """Constructor for NEAT"""
    def __init__(self):

    """
    Evaluation
    brief: evaluates generation of NEAT
    input: currentPopulation
    output:updatedPopulation 
    """
    def Evaluation(self,currentPopulation):


        return updatedPopulation

    """
    NextGeneration
    brief: takes updated Population and makes new population with mutations and crossbreeding
    input: oldPopulation
    output:newPopulation 
    """
    def NextGeneration(self,oldPopulation):

        return newPopulation
