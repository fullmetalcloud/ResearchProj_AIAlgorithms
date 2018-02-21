"""
/**** ConnectionGene ****/
By: Edward Guevara 
References:  Sources
Description: connection between two nodes and its innovation number (unique)
"""

# Neuron Gene shows unique id and what type of gene. 1 = input, 2 = output, 3 or more = hidden
# Node Gene array is an implicit dictionary of node neurons
# e.g. "neuron['1'] = 1" states neuron 1 is an input neuron

class ConnectionGene():
    """Constructor for ConnectionGene"""
    def __init__(self,n1, n2, innovationNum, mutated=None):
        self.n1 = n1                            # incoming node
        self.n2 = n2                            # receiving node
        self.innovationNum = innovationNum      # innovation connection number (unique)
        self.mutated = mutated                  # innovation numbers that mutated this connection (done by node mutation)
