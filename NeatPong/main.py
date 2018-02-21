import os
import time
from multiprocessing import Manager

import gym
import tensorflow as tf

from NEAT.neatq import NEATPong, NeatTest, NeatTestPoleBalance
from NEAT.plot import *
from NEAT.parameters import POP_SIZE

def main():
    # for testing
    render = False
    render_mod = 100

    neat_test = True

    # hyperparameters
    numInputs = 6400
    numOutputs = 1
    height = 80
    width = 80
    reward = []
    running_reward = None

    # location of saved files (networks, speciation graphs, videos)
    directory = os.getcwd() + "/tmp/"

    # parameters for monitoring and recording games
    episode_number_NEAT = 0
    time_neat, reward_neat = [0], [0]
    record_rate_NEAT = 1

    sortedPop = []

    if neat_test:
        test_agent = NeatTestPoleBalance()
        test_acc = None

        # file creation of Pole Balance NEAT
        filePB = open(directory + 'PoleBalance', 'w')
        filePB.write('episode,runningscore\n')
        filePB.flush()

        while test_acc is None or test_acc < 195:
            test_agent.PoleEvaluation()
            ordered_pop = sorted(test_agent.population, key=lambda g: g.answer, reverse=True)
            print(list(k.answer for k in ordered_pop[:100]))
            highest = ordered_pop[0].answer
            if POP_SIZE >= 100:
                test_acc = np.mean(list(k.answer for k in ordered_pop[:100]))
            else:
                test_acc = np.mean(list(k.answer for k in ordered_pop[:]))
            print('Best Answer: %f' % highest)
            print('Running Answer: %f' % test_acc)
            filePB.write('%i,%.5f\n' % (test_agent.numEpisodes, test_acc))
            filePB.flush()

    # openAI env init
    envNEAT = []
    for i in range(0, POP_SIZE):
        env = gym.make('Pong-v0')

        if 0 == i:
            # openAI env recording videos init
            env = gym.wrappers.Monitor(env,
                                       directory=directory,
                                       force=True,
                                       # video_callable=lambda episode_id: True)
                                       video_callable=lambda episode_id: 0 == episode_number_NEAT % record_rate_NEAT)
        envNEAT.append(env)
    # multiprocessing manager for sharing variables
    manager = Manager()

    # class creation of NEAT Agent
    neat_agent = NEATPong(numInputs,
                    numOutputs,
                    envNEAT,
                    render,
                    render_mod,
                    manager=manager)

    # file creation of NEAT
    fileNEAT= open(directory + 'NEATInfo', 'w')
    fileNEAT.write('episode,time,bestscore\n')

    try:
        while True:
            # evaluate NEAT algorithm
            episode_number_NEAT = neat_agent.numEpisodes
            start = time.time()
            neat_agent.PongEvaluation()
            end = time.time()
            # for genome in neat_agent.population:
            #     reward.append(genome.answer)
            # sortedPop = [x for _, x in sorted(zip(reward, neat_agent.population), key=lambda pair: pair[0], reverse=True)]
            # for i in sortedPop:
            #     print(i.answer)
            sortedPop = neat_agent.population[:]
            sortedPop.sort(key=lambda g: g.answer, reverse=True)
            running_reward = sortedPop[0].answer if running_reward is None else 0.99 * running_reward + 0.01 * sortedPop[0].answer
            print('Best Answer: %f' % sortedPop[0].answer)
            print("Neat Algorithm Time: " + str(end - start))
            print("Neat Algorithm Reward: %f" % running_reward)
            fileNEAT.write('%i,%.3f,%i\n' % (neat_agent.numEpisodes, (end - start), sortedPop[0].answer))
            fileNEAT.flush()
            time_neat.append(end - start)
            reward_neat.append(sortedPop[0].answer)
            if sortedPop[0].answer >= 0:
                print('\n\n<<<<< I WON >>>>>\n\n')
            # if sortedPop:
            #     ShowNEATNeuralNetwork(sortedPop[0], neat_agent.nodeGeneArray, directory, neat_agent.numEpisodes)
            if 0 == episode_number_NEAT % record_rate_NEAT and episode_number_NEAT != 0:
                ShowSpeciesChart(neat_agent.recordedSpecies, neat_agent.numSpecies, directory, neat_agent.numEpisodes)

    except KeyboardInterrupt:
        print('Interrupted')

if __name__ == '__main__':
    main()