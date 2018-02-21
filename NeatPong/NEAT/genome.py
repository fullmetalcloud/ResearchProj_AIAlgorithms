
import math
from NEAT.parameters import INIT_FUNC, SIG_MOD

"""
/**** Genome ****/
By: Edward Guevara 
References:  Sources
Description: array of connection genes with weights and activation list = neural network
"""
class Genome():
    """Constructor for Genome"""
    def __init__(self, geneArray, weights, activationList, genesMutated, speciesNum):
        self.geneArray = geneArray                      # array of connection genes,
        self.weights = weights                          # weights of connection genes
        self.activationList = activationList            # activation of connection gene
        self.genesMutated = genesMutated                # tracks the mutations that occurred in genome
        self.speciesNum = speciesNum                    # which species genome came from
        self.neuronArray = []                           # for evaluation of NN
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
    def GenomeEval(self, neuronArray, nodeGeneArray):
        # setup neuron array to evaluate each node
        self.neuronArray = neuronArray
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