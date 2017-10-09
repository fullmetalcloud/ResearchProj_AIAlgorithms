from neuralnet import NeuralNetwork
from convonn import ConvolutionNN
import gym
import time
import os

render = False

def main():
    # hyperparameters
    num_hidden_layer_neurons = 200
    height = 80
    width = 80
    directory = os.getcwd() + "/tmp/"
    episode_number = 0
    record_rate_NN = 100
    record_rate_convoNN = 100
    record_rate_NEAT = 100
    #initialize gym environment
    envNN = gym.make('Pong-v0')
    envConvoNN = gym.make('Pong-v0')
    envNEAT = gym.make('Pong-v0')
    envNN = gym.wrappers.Monitor(envNN, directory+"NN", force=True,
                               video_callable=lambda episode_id: 0 == episode_number % record_rate_NN)
    envConvoNN = gym.wrappers.Monitor(envConvoNN, directory+"ConvoNN", force=True,
                                 video_callable=lambda episode_id: 0 == episode_number % record_rate_convoNN)
    envNEAT = gym.wrappers.Monitor(envNEAT, directory+"NEAT", force=True,
                                 video_callable=lambda episode_id: 0 == episode_number % record_rate_NEAT)

    #initialize neural network algorithms
    nn = NeuralNetwork(num_hidden_layer_neurons, height * width)
    convo_nn = ConvolutionNN(num_hidden_layer_neurons, height, width)
    while True:
        print(episode_number)
        #evaluate Fully Connected Neural Network
        start = time.time()
        nn.eval(envNN, render)
        end = time.time()
        print("Neural Network Time: " + str(end - start))
        print("Neural Network Reward: " + str(nn.reward_sum))
        if nn.episode_num == 1000:
            record_rate_NN = 1000
        elif nn.episode_num == 10000:
            record_rate_NN = 2000

        #evaluate Convolutional Fully Connected Neural Network
        start = time.time()
        convo_nn.eval(envConvoNN, render)
        end = time.time()
        print("Convolutional Neural Network Time: " + str(end - start))
        print("Convolutional Neural Network Reward: " + str(convo_nn.reward_sum))
        episode_number = nn.episode_num
        if convo_nn.episode_num == 1000:
            record_rate_convoNN = 1000
        elif convo_nn.episode_num == 10000:
            record_rate_convoNN = 2000

main()