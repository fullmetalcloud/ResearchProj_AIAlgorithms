from NEAT.parameters import C1, C2, C3, GENE_SIZE_LIMIT
"""
/**** Species ****/
By: Edward Guevara 
References:  Sources
Description: group of genomes that match in topology and weights
"""
class Species():
    """Constructor for Species"""
    def __init__(self,genome, speciesNum):
        self.niche = []                                           # genomes grouped with species
        self.topGenome = genome                                   # best genome of the current and previous species
        self.speciesNum = speciesNum                              # species number
        self.niche.append(genome)                                 # append top genome to niche for newly made species
        self.adjustedFitness = 0                                  # adjusted fitness based on niche and fitness
        self.prevAdjustedFitness = 0                              # previous fitness to check for stalemate
        self.numOffspring = 0                                     # number of offspring made

    """
    Reset
    brief: deletes all but champions of Species
    input: 
    output:none 
    """
    def Reset(self):
        """TESTING AND OLD CODE IN ... """
        # check if species stalemate from previous generation
        # if abs(self.adjustedFitness - self.prevAdjustedFitness) < 0.001:
        #     self.stalemate += 1
        # else:
        #     self.stalemate = 1

        # arrGenome = "["
        # print("SPECIES NUM: %i" % self.speciesNum)
        # for g in self.niche:
        #     for gene in g.geneArray:
        #         arrGenome += str(gene.n1)
        #         arrGenome += ", "
        #         arrGenome += str(gene.n2)
        #         arrGenome += "; "
        #     print(arrGenome + "]")
        #     arrGenome = "["
        # print("SPECIES TOP GENOME: ")
        # for gene in self.topGenome.geneArray:
        #     arrGenome += str(gene.n1)
        #     arrGenome += ", "
        #     arrGenome += str(gene.n2)
        #     arrGenome += "; "
        # print(arrGenome + "]")
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
        # reevaluate fitness of all genomes in niche based on size of niche
        # helps adjust for overpopulation
        if self.niche:
            for genome in self.niche:
                genome.fitness = genome.fitness/len(self.niche)
                self.adjustedFitness+= genome.fitness
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
        excess = abs(len(newGenomeArray) - len(topGeneArray))

        # gene size is the biggest genome
        totalGenes = len(newGenomeArray) if (len(newGenomeArray) > len(topGeneArray)) else len(topGeneArray)
        if totalGenes < GENE_SIZE_LIMIT:
            totalGenes = 1

        # evaluation equation
        differenceVal = c1 * excess / totalGenes + c2 * disjoint / totalGenes + c3 * totalDiffWeights / totalGenes

        # check if difference is under or over threshold
        if speciesThreshold > differenceVal:
            return True, differenceVal
        else:
            return False, differenceVal

