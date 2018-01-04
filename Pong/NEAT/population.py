from NEAT.genome import *
from NEAT.gene import *
from NEAT.parameters import *
from NEAT.species import *

import random

"""
/**** Population ****/
By: Edward Guevara 
References:  Sources
Description: Population of Genomes, used for Species Evaluation
"""
class Population():
    """Constructor for Population"""
    def __init__(self, numInputs, numOutputs):
        self.population = []                    # population of genomes
        self.connGeneArray = []                 # innovation array of connected genes (all unique connections)
        self.nodeGeneArray = {}                 # innovation array of node genes/
                                                # dictionary of nodes associated to their respective layer
        self.speciesGroups = {}                 # species groups generated during evolution

        # initialize node gene array and connection gene array
        innovationNum = 0

        # all input node genes are in layer 1
        for i in range(0, numInputs):
            self.nodeGeneArray[i] = 1

        # all output node genes start at layer 2
        for j in range(numInputs, numInputs + numOutputs):
            self.nodeGeneArray[j] = 2

            # build connection genes between input nodes to output nodes
            for k in range(0, numInputs):
                connGene = ConnectionGene(k, j, innovationNum)
                self.connGeneArray.append(connGene)
                innovationNum += 1
        # setup population with minimal topology and weights
        self.ResetPopulation()

        # parameters for population evaluation and evolution
        self.numSpecies = 0
        self.deltaSpecies = SPECIES_THRESHOLD
        self.total = 0
        self.noChange =0
        self.numEpisodes = 0
        self.recordedSpecies = []

    """
    ResetPopulation
    brief: resets population with minimal topologies
    input: numInputs, numOutputs
    output:none 
    """
    def ResetPopulation(self):
        # empty current population
        self.population = []
        weights, activationList, genesMutated = [], [], []

        # build minimal topology for all genomes being added
        for q in range(0, POP_SIZE):
            for w in self.connGeneArray:

                # initialize weights, activation list, and gene mutations
                weights.append(self.InitWeight())
                activationList.append(True)
                genesMutated.append(False)

            # create Genome and add to population
            geneArray = self.connGeneArray[:]
            self.population.append(Genome(geneArray, weights, activationList, genesMutated, 1))
            weights, activationList, genesMutated = [], [], []
        return

    """
    AddSpecies
    brief: Adds a new species to population when genome is outside all current species' comparison
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

        # if species has 2 genomes to crossover
        if 1 < len(species.niche):
            g1, g2 = 0, 0
            newW, newArr, newActivationList, newGenesMutated = [], [], [], []

            # sort fitnesses of genomes
            genomeFitnesses = list(g.fitness for g in species.niche)

            # select genome to crossover
            genome1 = self.WeightedChoice(list(zip(species.niche, genomeFitnesses)))

            # cross species mutation or cross within niche
            if random.uniform(0, 1) <= CROSS_MUTATION_PROB:
                speciesX = random.choice(self.speciesGroups)

                # make sure species has a genome to crossover
                while not speciesX.niche:
                    speciesX = random.choice(self.speciesGroups)

                # sort niche and choose genome from species
                genomeFitnessesX = list(g.fitness for g in speciesX.niche)
                genome2 = self.WeightedChoice(list(zip(speciesX.niche, genomeFitnessesX)))
            else:
                # choose another genome from same niche
                genome2 = self.WeightedChoice(list(zip(species.niche, genomeFitnesses)))

            # get arrays from the two genomes
            gArr1 = genome1.geneArray
            gArr2 = genome2.geneArray

            # compare arrays based on innovation numbers of genes until either array
            # do not have anymore genes to compare
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
                    # randomly choose weight from genome1 or genome2
                    if random.uniform(0, 1) > 0.5:
                        newW.append(genome1.weights[g1])
                    else:
                        newW.append(genome2.weights[g2])

                    if (genome1.activationList[g1] and genome2.activationList[g2]):
                        # and random.randint(0, int(1 / ACTIVATION_PROB)) == 1:
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
            newGenome = Genome(newArr, newW, newActivationList, newGenesMutated, species.speciesNum)

        # if only one genome is in the niche, guarantee weight mutation
        elif 1 == len(species.niche):
            newGenome = species.niche[0]
            self.WeightMutation(newGenome)

        # Node, Connection, and Weight Mutations of new Genome
        if random.uniform(0, 1) <= NODE_PROB:
            self.NodeMutation(newGenome)
        if random.uniform(0, 1) <= CONN_PROB:
            self.ConnectionMutation(newGenome)
        if random.uniform(0, 1) <= WEIGHT_PROB:
            self.WeightMutation(newGenome)

        # sort gene array based on innovation number for next generation crossover
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

        # go through each species and crossover all genomes within species
        for species in self.speciesGroups.values():
            if species.niche:
                count = 0

                # breed amount of new offspring based on given value from speciation
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

        # decrease for more species and increase for less species
        if SPECIES_SIZE < len(self.speciesGroups):
            self.deltaSpecies += ADJUST_DELTA
        else:
            self.deltaSpecies -= ADJUST_DELTA

        # check boundaries of Delta
        if self.deltaSpecies < ADJUST_DELTA:
            self.deltaSpecies = ADJUST_DELTA
        elif self.deltaSpecies > ADJUST_DELTA:
            self.deltaSpecies = SPECIES_THRESHOLD
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
        totalDiffVal = 0

        # reset all species for sorting and evaluation
        for species in self.speciesGroups.values():
            species.Reset()

        # adjust species delta for more or less species creation
        self.CheckThreshold()

        # sort genomes in population into species
        for genome in self.population:

            # add new Species if species list is empty
            if not self.speciesGroups:
                self.speciesGroups[self.numSpecies] = Species(genome, self.numSpecies)
                self.numSpecies += 1
            else:
                check = False
                count = 0

                # compare genome to top genome of species
                while (count < len(self.speciesGroups.keys())) and not check:
                    species = self.speciesGroups[count]
                    check, diffVal = species.GenomeCompatibility(genome, self.deltaSpecies)
                    count+=1
                totalDiffVal += diffVal

                # if genome could not be sorted, a new species is found
                if not check:
                    self.AddSpecies(genome)

                # add genome to niche of found species
                else:
                    species.niche.append(genome)
                    check = False

        self.deltaSpecies = totalDiffVal/POP_SIZE
        # determine adjusted fitness of species for number of offspring
        for species in self.speciesGroups.values():
            species.AdjustedFitness()
            self.total+=species.adjustedFitness

        # stagnation of population based on total of adjusted fitness
        if abs(self.total-oldTotal) < 0.01:
            self.noChange+=1

        # if stagnation occurs, breed only top two species
        if self.noChange > NO_CHANGE_LIMIT:
            self.noChange = 0
            topTwo = sorted(self.speciesGroups.values(),
                                 key=lambda s: s.adjustedFitness, reverse=True)
            for species in self.speciesGroups.values():
                # determine number of offspring
                species.numOffspring = 0
                # keep top 20% in niche
                size = len(species.niche) * PERCENT_CHAMPS
                if size > 2:
                    species.niche = species.niche[:int(size) - 1]
            for s in topTwo[:2]:
                # determine number of offspring
                s.numOffspring = POP_SIZE / 2
                # sort niche for elimination
                s.niche.sort(key=lambda genome: genome.fitness, reverse=True)

        # determine number of children that can be made by species
        else:
            for species in self.speciesGroups.values():
                # determine number of offspring
                species.numOffspring = species.adjustedFitness/self.total*POP_SIZE

                # sort niche for elimination
                species.niche.sort(key=lambda genome: genome.fitness, reverse=True)

                # keep % in niche
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
        # determine weights using Xavier Initialization
        # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        xavier_var = 2/math.sqrt(len(self.nodeGeneArray.keys()))
        value = random.uniform(-INIT_WEIGHT_VALUE * xavier_var, INIT_WEIGHT_VALUE * xavier_var)
        return value

    """
    NodeMutation
    brief: Mutates a genome with a new node
    input: genome
    output:none 
    """

    def NodeMutation(self, genome):
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

            # get node genes
            n1 = array[chosenGene].n1
            n2 = array[chosenGene].n2

            # update layers if adding new layer
            if (self.nodeGeneArray[n2] - self.nodeGeneArray[n1] == 1):
                for n, layerVal in self.nodeGeneArray.items():
                    if layerVal >= self.nodeGeneArray[n2]:
                        self.nodeGeneArray[n] += 1

            # make new node/ add to node gene array as 1 layer behind the second node
            innovationNum = len(self.connGeneArray)
            newNode = max(self.nodeGeneArray.keys()) + 1
            self.nodeGeneArray[newNode] = self.nodeGeneArray[n2] - 1

            # add 2 new connection genes to connection gene array and to genome
            connGene1 = ConnectionGene(n1, newNode, innovationNum)
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

            # record the two new connection genes that resulted from mutated
            # connection gene
            self.connGeneArray[array[chosenGene].innovationNum].mutated = (innovationNum - 1, innovationNum)
            array[chosenGene].mutated = (innovationNum - 1, innovationNum)
        return

    """
    ConnectionMutation
    brief: Mutates a genome to add a new connection
    input: genome
    output:none 
    """

    def ConnectionMutation(self, genome):
        n1 = 0
        n2 = 0
        choice = False
        count = 0
        check = False
        array = genome.geneArray
        newConnGene = None
        while not choice and count < len(array):
            # radnomly choose two genes from genome array
            n1 = random.choice(array).n1
            n2 = random.choice(array).n2

            # make sure they are not the same genes
            if n1 == n2:
                choice = False
            else:
                # check all connection genes to make sure conn gene is not in the genome array
                for gene in array:
                    # check if connection gene is n1 and n2 and is activated
                    if gene.n1 == n1 and gene.n2 == n2 and genome.activationList[count]:
                        if (self.nodeGeneArray[n2] - self.nodeGeneArray[n1]) == 1:
                            choice = False
                            count += 1
                            break
                        else:
                            # deactivate gene if chosen and activated
                            genome.activationList[count] = False
                            return
                    else:
                        # chosen connection gene does not exist in genome array
                        choice = True
        # if all existing connection genes exist for all node genes (i.e. fully connected)
        if count == len(array):
            return

        # check if connection gene exists with 2 chosen node genes
        for connGene in self.connGeneArray:
            if connGene.n1 == n1 and connGene.n2 == n2:
                if not genome.activationList[count]:
                    # activate if exists in genome array
                    genome.activationList[count] = True
                    check = True
                    break
                else:
                    # add to genome array
                    newConnGene = connGene
                    innoNum = newConnGene.innovationNum
                    genome.geneArray.insert(innoNum, newConnGene)
                    genome.activationList.insert(innoNum, True)
                    genome.weights.insert(innoNum, self.InitWeight())
                    genome.genesMutated.insert(innoNum, False)
                    check = True
                    break
        # if brand new connection between two node genes, make new connection gene
        # and add to connection gene array with new innovation number
        if not check:
            newConnGene = ConnectionGene(n1, n2, self.connGeneArray[-1].innovationNum + 1)
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

    def WeightMutation(self, genome):
        # chance it can be a new weight
        if random.uniform(0, 1) <= NEW_WEIGHT_PROB:
            for i in range(0, random.randint(0, len(genome.weights))):
                genome.weights[i] = self.InitWeight()
        else:
        # adjust weight by standard deviation
            for i in range(0, random.randint(0,len(genome.weights))):
                genome.weights[i] += random.uniform(-WEIGHT_STDEV, WEIGHT_STDEV)
        return