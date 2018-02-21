
from NEAT.pong_image import *
from NEAT.population import *
import copy
import time
import gym
from multiprocessing import Process, Array, Manager, Pool

"""
/**** NEATPong ****/
By: Edward Guevara 
References:  Sources
Description: NEAT Algorithm Implementation using Pong
"""
class NEATPong(Population):
    """Constructor for NEAT"""
    def __init__(self, numInputs, numOutputs, env, render=False, render_mod=100, manager=None):
        self.env = env                                          # environment for OpenAI
        self.numInputs = numInputs                              # number of inputs that comes from OpenAI env
        self.render = render                                    # boolean to render/show game
        self.render_mod = render_mod                            # if render = True, mod when to show game
        self.recordedSpecies = []
        self.manager = manager                                # for parallelization, manager for sharing vars
        self.sharedList = manager.list(range(POP_SIZE))       # population list shared with processes
        super().__init__(numInputs, numOutputs)                 # general init for setting up neat algorithm

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


    """Policy Gradient Functions"""
    def discount_rewards(self,rewards, gamma):
        """ Actions you took 20 steps before the end result are less important to the overall result than an action 
        you took a step ago. This implements that logic by discounting the reward on previous actions based on how 
        long ago they were taken"""
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + rewards[t] + 1.0
            discounted_rewards[t] = running_add
        return discounted_rewards

    """
    ParallelEval
    brief: runs fitness function in parallel
    input: func, args
    output: p 
    """
    def ParallelEval(self,func, args):

        return

    """
    PongEvaluation
    brief: evaluates generation of NEAT
    input: 
    output: 
    """
    def PongEvaluation(self):
        # parallelization vars
        processes = []
        population = self.sharedList
        # population = self.population
        # localized function calls
        fitnessFunction = self.PongFitnessFunction
        speciation = self.Speciation
        crossover = self.Crossover
        sortPop = self.population.sort

        # process: EVALUATION -> SPECIATION -> CROSSOVER -> EVALUATION -> SPECIATION...

        # if starting episode, run initial fitness evaluation
        if self.numEpisodes == 0:
            """Parallelization"""
            # start = time.time()
            # for i, genome in enumerate(self.population):
            #     if 0 == i:
            #         fitnessFunction(population, self.nodeGeneArray, genome, i, self.env[i])
            #     else:
            #         p = Process(target=fitnessFunction, args=(population, self.nodeGeneArray, genome, i, self.env[i]))
            #         processes.append(p)
            #         if i % 10 == 0 or i == len(self.population)-1:
            #             for p in processes:
            #                 p.start()
            #             for p in processes:
            #                 p.join()
            #             processes = []
            # print("PARALLEL TIME: %.12f" % (time.time() - start))
            # self.population = list(population[:])
            """Sequential"""
            start = time.time()
            for i, genome in enumerate(self.population):
                fitnessFunction(population, self.nodeGeneArray, genome, i, self.env[i])
            print("SEQUENTIAL TIME: %.12f" % (time.time()-start))
        self.numEpisodes += 1

        # Speciation
        start = time.time()
        speciation()
        print("SPECIATION TIME: %.12f" % (time.time()-start))

        # Crossover
        start = time.time()
        crossover()
        print("CROSSOVER TIME: %.12f" % (time.time() - start))

        # sort population
        sortPop(key=lambda g: g.fitness, reverse=True)
        # start = time.time()
        # for i, genome in enumerate(self.population):
        #     if 0 == i:
        #         fitnessFunction(population, self.nodeGeneArray, genome, i, self.env[i])
        #     else:
        #         p = Process(target=fitnessFunction, args=(population, self.nodeGeneArray, genome, i, self.env[i]))
        #         processes.append(p)
        #         if i % 10 == 0 or i == len(self.population)-1:
        #             for p in processes:
        #                 p.start()
        #             for p in processes:
        #                 p.join()
        #             processes = []
        # print("PARALLEL TIME: %.12f" % (time.time() - start))
        # self.population = list(population[:])
        # Apply fitness evaluation
        start = time.time()
        for i, genome in enumerate(self.population):
            fitnessFunction(population, self.nodeGeneArray, genome, i, self.env[i])
        print("SEQUENTIAL TIME: %.12f" % (time.time() - start))

        # record current generation of species for image evaluation
        self.recordedSpecies.append(copy.deepcopy(self.speciesGroups))

        # show information needed to be seen
        print('\n')
        print(self.numEpisodes, len(self.speciesGroups))
        print(len(self.nodeGeneArray), len(self.connGeneArray))
        # arrGenome = "["
        # g = self.population[0]
        # print(g.answer)
        # print(g.tests)
        # for gene in g.geneArray:
        #     arrGenome += str(gene.n1)
        #     arrGenome += ", "
        #     arrGenome += str(gene.n2)
        #     arrGenome += "; "
        # print(arrGenome + "]")
        return

    """
    PongFitnessFunction
    brief: evaluates genome for Pong
    input: env, inputDimensions, nodeGeneArray, generation, genome, render, render_mod
    output:none 
    """

    def PongFitnessFunction(self, population, nodeGeneArray, genome, i, env):
        # print("Genome running: %i" % i)
        # start = time.time()

        # initial vars and function calls
        observation = env.reset()
        rewardSum = 1
        done = False
        prev_processed_observations = None
        eval = genome.GenomeEval
        maxNode = max(nodeGeneArray, key=nodeGeneArray.get)
        process = preprocess_observations
        ep_reward = []

        # setup neuron array evaluation
        neuronArray = [0 for _ in range(len(nodeGeneArray))]

        # determine if game is to be rendered/shown
        render = i == 0 and self.render

        # play until game ends
        while not done:
            if render:
                env.render()

            # preprocess image
            neuronArray[:self.numInputs], prev_processed_observations = process(observation,
                                                                          prev_processed_observations,
                                                                          self.numInputs)
            # evaluate neural network
            eval(neuronArray[:], nodeGeneArray)

            #determine action
            up_probability = genome.neuronArray[maxNode]
            action = self.chooseAction(up_probability)

            # carry out the chosen action
            observation, reward, done, info = env.step(action)
            rewardSum += reward
            ep_reward.append(reward)

            # see here: http://cs231n.github.io/neural-networks-2/#losses
            fake_label = 1 if action == 2 else 0
            loss_function_gradient = fake_label - up_probability

        # TODO: determine fitness function and Apply Q-Learning and policy gradient eval
        genome.fitness = np.square((22.0 + rewardSum)/43.0)         # evaluated to equalize
                                                                    # score from (-21)->21 to 1->43
        # genome.fitness = np.mean(self.discount_rewards(np.array(ep_reward), 0.99))
        genome.answer = rewardSum
        population[i] = genome
        # end = time.time()
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
    def __init__(self, manager):
        numInputs = 3                                           # predetermined number of inputs for XOR
        numOutputs = 1                                          # predetermined number of outputs for XOR
        self.manager = manager                                  # manager for multiprocessing
        self.sharedList = self.manager.list(range(POP_SIZE))    # shared list for multiprocessing
        super().__init__(numInputs, numOutputs)                 # init for neat algorithm

    """
    XORFitnessFunction
    brief: evaluates population for XOR example
    input: genome
    output:none 
    """

    def XORFitnessFunction(self, nodeGeneArray, genome):
        # initial vars
        total = 0
        fitness = 0
        genome.tests = []
        input = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        output = [0, 1, 1, 0]
        neuronArray = [0 for _ in range(len(nodeGeneArray))]

        # evaluate genomes with XOR test
        for model in range(0, 4):
            neuronArray[:len(input[model])] = input[model]
            genome.GenomeEval(neuronArray[:], nodeGeneArray)
            ans = genome.neuronArray[max(nodeGeneArray, key = nodeGeneArray.get)]
            total += 1 - abs(output[model] - ans)

            # fitness is the (expected - output)^2
            fitness += 1 - np.square(output[model] - ans)
            genome.tests.append(ans)

        # get average of answers of XOR and fitness
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
        # initial function calls
        speciation = self.Speciation
        crossover = self.Crossover
        sortPopulation = self.population.sort
        fitnessFunction = self.XORFitnessFunction

        # # parallelization vars
        # population = self.sharedList
        # processes = []
        check = False
        test = 0
        arrGenome = ""
        # process: EVALUATION -> SPECIATION -> CROSSOVER -> EVALUATION -> SPECIATION...
        for j, genome in enumerate(self.population):
            fitnessFunction(self.nodeGeneArray, genome)
        while not check:
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

            # Speciation
            start = time.time()
            speciation()
            # end = time.time()
            # print("SPECIATION TIME: %.12f" % (end-start))

            # Crossover
            # start = time.time()
            crossover()
            # end = time.time()
            # print("CROSSOVER TIME: %.12f" % (end-start))

            # fitness evaluation
            # start = time.time()
            for genome in self.population:
                fitnessFunction(self.nodeGeneArray, genome)
            # end = time.time()
            # print("EVALUATION TIME: %.12f" % (end-start))

            # sort population for best answer
            sortPopulation(key=lambda g: g.answer, reverse=True)

            # check if any genome meets desired accuracy
            for genome in self.population:
                if genome.answer > EXPECTED_ACCURACY:
                    check = True
                    break

            # record current generation of species for image evaluation
            self.recordedSpecies.append(copy.deepcopy(self.speciesGroups))

            # reset population to avoid stagnation of population
            if self.numEpisodes % POP_RESET_LIMIT == POP_RESET_LIMIT-1:
                self.ResetPopulation()
        return

"""
/**** NeatTestPoleBalance ****/
By: Edward Guevara 
References:  Sources
Description: Tests the Neat Algorithm using the Pole Balancing Example
"""
class NeatTestPoleBalance(Population):
    """Constructor for NeatTestPoleBalance"""
    def __init__(self):
        numInputs = 4  # predetermined number of inputs for PoleBalance
        numOutputs = 1  # predetermined number of outputs for PoleBalance
        self.render = False  # boolean to render/show game
        self.render_mod = 100
        self.env = []
        for i in range(0, POP_SIZE):
            self.env.append(gym.make('CartPole-v0'))
        self.manager = Manager()  # manager for multiprocessing
        self.sharedList = self.manager.list(range(POP_SIZE))  # shared list for multiprocessing
        super().__init__(numInputs, numOutputs)

    """
    PoleFitnessFunction
    brief: Fitness Function for Pole balancing
    input: nodeGeneArray, genome
    output: 
    """
    def PoleFitnessFunction(self, population, nodeGeneArray, genome, i, env):
        # initial vars and function calls
        observation = env.reset()
        rewardSum = 0
        done = False
        eval = genome.GenomeEval
        maxNode = max(nodeGeneArray, key=nodeGeneArray.get)

        # setup neuron array evaluation
        neuronArray = [0 for _ in range(len(nodeGeneArray))]

        # determine if game is to be rendered/shown
        render = self.render and self.numEpisodes % self.render_mod == 0

        # play until game ends
        while not done:
            if render:
                env.render()

            # add observations to neuron array
            neuronArray[:len(observation)] = observation

            # evaluate neural network
            eval(neuronArray[:], nodeGeneArray)

            # determine action
            up_prob = genome.neuronArray[maxNode]
            action = 0 if up_prob < np.random.uniform() else 1

            # carry out the chosen action
            observation, reward, done, info = env.step(action)
            rewardSum += reward

        # TODO: determine fitness function and Apply Q-Learning and policy gradient eval
        genome.answer = genome.fitness = rewardSum
        population[i] = genome
        # end = time.time()
        # print("GENOME %i Time: %.12f" % (i, end-start))
        # print("GENOME %i Fitness: %.12f" % (i, population[i].answer))
        return

    """
    PoleEvaluation
    brief: evaluation of pole balancing example
    input: 
    output: 
    """
    def PoleEvaluation(self):
        # parallelization vars
        processes = []
        # population = self.sharedList
        population = self.population
        # localized function calls
        fitnessFunction = self.PoleFitnessFunction
        speciation = self.Speciation
        crossover = self.Crossover

        # process: EVALUATION -> SPECIATION -> CROSSOVER -> EVALUATION -> SPECIATION...

        # if starting episode, run initial fitness evaluation
        if self.numEpisodes == 0:
            """Parallelization"""
            # start = time.time()
            #
            # for i, genome in enumerate(self.population):
            #     p = Process(target=fitnessFunction, args=(population, self.nodeGeneArray, genome, i, self.env[i]))
            #     processes.append(p)
            #     if i % 10 == 0 or i == len(self.population)-1:
            #         for p in processes:
            #             p.start()
            #         for p in processes:
            #             p.join()
            #         processes = []
            # print("PARALLEL TIME: %.12f" % (time.time() - start))
            # self.population = list(self.sharedList[:])
            processes = []
            """Sequential"""
            start = time.time()
            for i, genome in enumerate(self.population):
                fitnessFunction(population, self.nodeGeneArray, genome, i, self.env[i])
            print("SEQUENTIAL TIME: %.12f" % (time.time()-start))
        self.numEpisodes += 1

        # Speciation
        start = time.time()
        speciation()
        print("SPECIATION TIME: %.12f" % (time.time() - start))

        # Crossover
        start = time.time()
        crossover()
        print("CROSSOVER TIME: %.12f" % (time.time() - start))

        # sort population
        population = sorted(population, key=lambda g: g.answer, reverse=True)
        # start = time.time()
        # for i, genome in enumerate(self.population):
        #     p = Process(target=fitnessFunction, args=(population, self.nodeGeneArray, genome, i, self.env[i]))
        #     processes.append(p)
        #     if i % 10 == 0 or i == len(self.population)-1:
        #         for p in processes:
        #             p.start()
        #         for p in processes:
        #             p.join()
        #         processes = []
        # print("PARALLEL TIME: %.12f" % (time.time() - start))
        # self.population = list(self.sharedList[:])

        # Apply fitness evaluation
        start = time.time()
        for i, genome in enumerate(self.population):
            fitnessFunction(population, self.nodeGeneArray, genome, i, self.env[i])
        print("SEQUENTIAL TIME: %.12f" % (time.time() - start))

        # show information needed to be seen
        print(self.numEpisodes, len(self.speciesGroups))
        print(len(self.nodeGeneArray), len(self.connGeneArray))
        # arrGenome = "["
        # g = self.population[0]
        # print(g.answer)
        # print(g.tests)
        # for gene in g.geneArray:
        #     arrGenome += str(gene.n1)
        #     arrGenome += ", "
        #     arrGenome += str(gene.n2)
        #     arrGenome += "; "
        # print(arrGenome + "]")
        return