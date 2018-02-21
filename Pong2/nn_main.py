import os
import time
import tensorflow as tf
import gym
from pong_agents import Pong_PoliGrad_Agent

def nn_main():
    # hyperparameters
    num_layer_neurons_NN = [6400, 200, 1]

    # location of saved files (networks, speciation graphs, videos)
    directory = os.getcwd() + "/tmp/NN"

    # parameters for monitoring and recording games
    running_reward_NN = None
    prev_running_reward_NN = 0
    best_reward_NN = None
    episode_number_NN = 0
    time_nn, reward_nn = [0], [0]
    record_rate_NN = 100

    render = False

    # tensorflow session init
    sess_nn = tf.Session()

    # openAI env init
    envNN = gym.make('Pong-v0')

    # openAI env recording videos init
    envNN = gym.wrappers.Monitor(envNN,
                                 directory,
                                 force=True,
                                 video_callable=lambda episode_id: 0 == episode_number_NN % record_rate_NN)

    # file creation of NN
    fileNN = open(directory + '/NNInfo', 'w')
    fileNN.write('episode,time,score\n')
    # fileBatchNN = open(directory + 'NN/NNBatchInfo', 'w')
    # fileBatchNN.write('batch,batch_loss,batch_entropy\n')

    # class creation of NN Agent
    nn = Pong_PoliGrad_Agent(num_layer_neurons_NN,
                             sess_nn,
                             envNN,
                             directory=directory)


    while True:
        try:
            # Pong Game
            # evaluate Fully Connected Neural Network
            episode_number_NN = nn.episode_num
            print(episode_number_NN)
            start = time.time()
            nn.pong_eval()
            end = time.time()
            print("NN Reward: %i Time: %.3f" % (nn.reward_sum, end - start))
            fileNN.write('%i,%.3f,%i\n' % (episode_number_NN,
                                           (end - start),
                                           nn.reward_sum))
            fileNN.flush()
            # fileBatchNN.write('%i,%.5f,%.8f\n' % (nn.buffer_update,
            #                                       sum(nn.batch_loss) / len(nn.batch_loss),
            #                                       sum(nn.batch_entropy) / len(nn.batch_entropy)))
            # fileBatchNN.flush()

            # score and improvement records
            running_reward_NN = nn.reward_sum if running_reward_NN is None \
                else running_reward_NN * 0.99 + nn.reward_sum * 0.01
            if best_reward_NN is None:
                best_reward_NN = nn.reward_sum
            elif nn.reward_sum > best_reward_NN:
                best_reward_NN = nn.reward_sum

            # show average score, running score, and average time
            if episode_number_NN % nn.poli_grad_nn.batch_size == 0:

                print(
                    'NN World Perf: Episode %f. mean reward: %f. diff: %f time: %.4f Top Score: %i' % (
                        nn.episode_num,
                        running_reward_NN,
                        running_reward_NN - prev_running_reward_NN,
                        sum(time_nn) / len(time_nn),
                        best_reward_NN))
                prev_running_reward_NN = running_reward_NN
                time_nn, reward_nn = [], []
            else:
                time_nn.append(end - start)
                reward_nn.append(nn.reward_sum)
            if episode_number_NN == 1000:
                # record_rate_NN = 1000
                nn.render_mod = 1000
            elif episode_number_NN == 10000:
                # record_rate_NN = 2000
                nn.render_mod = 2000
        except KeyboardInterrupt:
            fileNN.close()

nn_main()