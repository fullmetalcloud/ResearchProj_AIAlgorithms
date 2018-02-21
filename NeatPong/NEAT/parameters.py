
"""
NeatTest Parameters
"""
# POP_SIZE = 500                               # Population size
# SPECIES_THRESHOLD = 1                      # Initial Species delta if need more or less species
# ADJUST_DELTA = 0.02                         # change in Species Delta (increase for more and vice versa)
# GENE_SIZE_LIMIT = 6                      # Gene size limit before gene size is taken into account
# PERCENT_CHAMPS = 0.1                       # percent of genomes that are saved
# SPECIES_SIZE = 7                            # Number of desired species before delta declination
# NO_CHANGE_LIMIT = 10                       # Limit of generations a population does not change in fitness
# POP_RESET_LIMIT = 2000                      # number of episodes before full reset
#
# ACTIVATION_PROB = 0.25                      # chance activation is second parent genome
# NODE_PROB = 0.02                             # chance node mutation occurs to new genome
# CONN_PROB = 0.05                             # chance connection mutation occurs to new genome
# WEIGHT_PROB = 0.8                           # chance Weight mutation occurs to new genome
# NEW_WEIGHT_PROB = 0.2                       # chance weight is changed completely
# CROSS_MUTATION_PROB = 0.01                  # chance genome crosses over with another genome from another species
#
# SIG_MOD = 4.9                                 # adjustment to sigmoid function
# INIT_WEIGHT_VALUE = 1                       # largest starting weight of connection
# WEIGHT_STDEV = 0.03                         # deviation and perturbation of weight
# C1, C2, C3 = 1, 1, 0.3                      # constants of genome compatibility
# INIT_FUNC = 'Sigmoid'                          # initial function of layer evaluation (default is Sigmoid)
#                                             # check list in layerEvaluation for names of functions
#                                             # last layer is always evaluated with Sigmoid function
#
# EXPECTED_ACCURACY = 0.99                     # (NEATTest only) expected accuracy of XOR test

"""
NeatPoleBalanceTest Parameters
"""

POP_SIZE = 150                               # Population size
SPECIES_THRESHOLD = 1                      # Initial Species delta if need more or less species
ADJUST_DELTA = 0.3                          # change in Species Delta (increase for more and vice versa)
GENE_SIZE_LIMIT = 6                      # Gene size limit before gene size is taken into account
PERCENT_CHAMPS = 0.05                       # percent of genomes that are saved
SPECIES_SIZE = 10                            # Number of desired species before delta declination
NO_CHANGE_LIMIT = 15                       # Limit of generations a population does not change in fitness
POP_RESET_LIMIT = 2000                      # number of episodes before full reset
STAGNATION_DEFAULT = 0.01                    # value that defines if population has stagnated

ACTIVATION_PROB = 0.25                      # chance activation is second parent genome
NODE_PROB = 0.03                             # chance node mutation occurs to new genome
CONN_PROB = 0.05                             # chance connection mutation occurs to new genome
WEIGHT_PROB = 0.8                           # chance Weight mutation occurs to new genome
NEW_WEIGHT_PROB = 0.2                       # chance weight is changed completely
CROSS_MUTATION_PROB = 0.01                  # chance genome crosses over with another genome from another species

SIG_MOD = 4.9                                 # adjustment to sigmoid function
INIT_WEIGHT_VALUE = 1                       # largest starting weight of connection
WEIGHT_STDEV = 0.03                          # deviation and perturbation of weight
WEIGHT_ADJUST = 1                         # Adjustment for Weight Mutation
C1, C2, C3 = 1, 1, 0.3                      # constants of genome compatibility
INIT_FUNC = 'Sigmoid'                          # initial function of layer evaluation (default is Sigmoid)
                                            # check list in layerEvaluation for names of functions
                                            # last layer is always evaluated with Sigmoid function



"""
NeatPong Parameters
"""

# POP_SIZE = 100                              # Population size
# SPECIES_THRESHOLD = 50                      # Initial Species delta if need more or less species
# ADJUST_DELTA = 3                          # change in Species Delta (increase for more and vice versa)
# GENE_SIZE_LIMIT = 6600                      # Gene size limit before gene size is taken into account
# PERCENT_CHAMPS = 0.05                       # percent of genomes that are saved
# SPECIES_SIZE = 100                            # Number of desired species before delta declination
# NO_CHANGE_LIMIT = 10                       # Limit of generations a population does not change in fitness
# POP_RESET_LIMIT = 20000                      # number of episodes before full reset
# STAGNATION_DEFAULT = 0.1                    # value that defines if population has stagnated
#
# ACTIVATION_PROB = 0.25                      # chance activation is second parent genome
# NODE_PROB = 0.03                             # chance node mutation occurs to new genome
# CONN_PROB = 0.5                             # chance connection mutation occurs to new genome
# WEIGHT_PROB = 0.9                           # chance Weight mutation occurs to new genome
# NEW_WEIGHT_PROB = 0.1                       # chance weight is changed completely
# CROSS_MUTATION_PROB = 0.1                  # chance genome crosses over with another genome from another species
#
# SIG_MOD = 4.9                                 # adjustment to sigmoid function
# INIT_WEIGHT_VALUE = 1                       # largest starting weight of connection
# WEIGHT_STDEV = 0.1                          # deviation and perturbation of weight
# WEIGHT_ADJUST = 0.1                         # Adjustment for Weight Mutation
# C1, C2, C3 = 1, 1, 0.3                      # constants of genome compatibility C1 = Node, C2 = Conn, C3 = avg weight
# INIT_FUNC = 'Sigmoid'                          # initial function of layer evaluation (default is Sigmoid)
#                                             # check list in layerEvaluation for names of functions
#                                             # last layer is always evaluated with Sigmoid function
#
