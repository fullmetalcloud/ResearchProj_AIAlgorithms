from neuralnet import NeuralNetwork
from convonn import ConvolutionNN
import tensorflow as tf
import gym
import time
import os

render = False

def main():
    # hyperparameters
    num_hidden_layer_neurons = 200
    num_output_neurons = 1
    height = 80
    width = 80
    directory = os.getcwd() + "/tmp/"

    render_mod = 10

    #parameters for monitoring and recording games
    episode_number_NN = 0
    episode_number_ConvoNN = 0
    episode_number_NEAT = 0
    record_rate_NN = 100
    record_rate_convoNN = 100
    record_rate_NEAT = 100

    sess_1 = tf.Session()
    sess_2 = tf.Session()
    #initialize gym environment
    envNN = gym.make('Pong-v0')
    envConvoNN = gym.make('Pong-v0')
    envNEAT = gym.make('Pong-v0')
    envNN = gym.wrappers.Monitor(envNN, directory+"NN", force=True,
                               video_callable=lambda episode_id: 0 == episode_number_NN % record_rate_NN)
    envConvoNN = gym.wrappers.Monitor(envConvoNN, directory+"ConvoNN", force=True,
                                 video_callable=lambda episode_id: 0 == episode_number_ConvoNN % record_rate_convoNN)
    envNEAT = gym.wrappers.Monitor(envNEAT, directory+"NEAT", force=True,
                                 video_callable=lambda episode_id: 0 == episode_number_NEAT % record_rate_NEAT)

    #initialize neural network algorithms
    nn = NeuralNetwork(height * width, num_hidden_layer_neurons, num_output_neurons, batch_size=1)
    convo_nn = ConvolutionNN(num_hidden_layer_neurons,num_output_neurons, height, width, batch_size=1)
    while True:
        episode_number_NN = nn.episode_num
        print(episode_number_NN)
        #evaluate Fully Connected Neural Network
        start = time.time()
        nn.eval(envNN, sess_1, render)
        end = time.time()

        print("Neural Network Time: " + str(end - start))
        print("Neural Network Reward: " + str(nn.reward_sum))
        if episode_number_NN == 1000:
            record_rate_NN = 1000
        elif episode_number_NN == 10000:
            record_rate_NN = 2000

        #evaluate Convolutional Fully Connected Neural Network
        episode_number_ConvoNN = convo_nn.episode_num
        start = time.time()
        convo_nn.eval(envConvoNN, sess_2, render)
        end = time.time()
        print("Convolutional Neural Network Time: " + str(end - start))
        print("Convolutional Neural Network Reward: " + str(convo_nn.reward_sum))
        if convo_nn.episode_num == 1000:
            record_rate_convoNN = 1000
        elif convo_nn.episode_num == 10000:
            record_rate_convoNN = 2000

main()