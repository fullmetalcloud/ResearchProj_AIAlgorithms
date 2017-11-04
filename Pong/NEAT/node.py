from enum import Enum

class NodeType(Enum):
    NEURON = 0
    SENSOR = 1

class NodePlace(Enum):
    HIDDEN = 0
    INPUT = 1
    OUTPUT = 2
    BIAS = 3
class FuncType(Enum):
    SIGMOID = 0
    RELU = 1


class NNode():
    def __init__(self, ntype, nodeid):
        self.active_flag = False

