import matplotlib.pyplot as pyplot
import numpy as np

fig, ax = pyplot.subplots()

"""
ShowNEATNeuralNetwork
brief: graph neural network
input: genome
output:none 
"""
def ShowNEATNeuralNetwork(genome, nodeGeneArray, directory, test_num):
    # dictionary of layers for nodes
    layers = {}

    # clear previous image
    pyplot.clf()

    # each layer is an array of nodes
    for v in set(nodeGeneArray.values()):
        layers[v] = []

    # append nodes to layers based on NodeGeneArray
    for k, v in nodeGeneArray.items():
        layers[v].append(k)

    # determine layer with most nodes for spacing of nodes
    largestLayerSize = max(len(x) for x in layers.values())

    for i, j in enumerate(genome.geneArray):
        # get nodes for each connection gene in geneArray
        neuron1 = j.n1
        neuron2 = j.n2

        # determine layer of given nodes
        layerN1 = layers[nodeGeneArray[neuron1]]
        layerN2 = layers[nodeGeneArray[neuron2]]

        # determine spot of node on layer
        n1 = layerN1.index(neuron1)
        n2 = layerN2.index(neuron2)

        # determine type of node (green=input, black=hidden or output)
        type1 = nodeGeneArray[neuron1]
        type2 = nodeGeneArray[neuron2]

        # determine actual spot on y axis of graph
        posN1 = n1 * largestLayerSize / len(layerN1) + largestLayerSize / len(layerN1) / 2
        posN2 = n2 * largestLayerSize / len(layerN2) + largestLayerSize / len(layerN2) / 2

        # determine x and y coords of two points of line
        line_y = (posN1, posN2)
        line_x = (type1, type2)

        # if connection gene is active
        if genome.activationList[i]:

            # bigger the weight, thicker the line
            if genome.weights[i] > 0:
                pyplot.plot(line_x, line_y, linewidth=genome.weights[i], color='blue')
            else:
                pyplot.plot(line_x, line_y, linewidth=genome.weights[i], color='red')

        # determine color of input node
        if type1 == 1:
            pyplot.plot(type1, posN1, 'o', color='green')
        else:
            pyplot.plot(type1, posN1, 'o', color='black')

        # plot output node
        pyplot.plot(type2, posN2, 'o', color='black')
    # save and show figure
    pyplot.savefig(directory + 'NetworkCharts/' + str(test_num) +'.png', bbox_inches='tight')
    # pyplot.show()
    return

"""
ShowSpeciesChart
brief: Plots number of species per episode
input: recordedSpecies
output:none 
"""
def ShowSpeciesChart(recordedSpecies, numberOfSpecies,directory, test_num):
    print('Species Chart making')
    # dictionary of lines of species
    speciesCurves = {}

    # clear previous image and initialize plot
    pyplot.clf()
    pyplot.title("SPECIATION")
    pyplot.xlabel("GENERATIONS")
    pyplot.ylabel("POPULATION")
    pyplot.xlim(0, len(recordedSpecies))
    x = []

    # go through each generation of recorded species over time
    for i, generation in enumerate(recordedSpecies):
        # record generation
        x.append(i)
        y = 0

        # record number of offspring for species at generation x
        for s in range(0, numberOfSpecies):
            if s in generation:
                y += generation[s].numOffspring
            if s not in speciesCurves:
                speciesCurves[s] = []
            speciesCurves[s].append([i,y])
    y = np.zeros(len(recordedSpecies))

    # go through created x,y plot and add fill in color
    for k, s in speciesCurves.items():
        prevY = y
        y = []
        for plot in s:
            y.append(plot[1])
        pyplot.fill_between(x, prevY, y, label=("Spec %i" % k))

    # add legend when done
    pyplot.legend(bbox_to_anchor=(1, 1), loc=2)

    # save and show figure
    pyplot.savefig("%sSpeciesCharts/%i.png" % (directory,test_num), bbox_inches='tight')
    # pyplot.show()

    return