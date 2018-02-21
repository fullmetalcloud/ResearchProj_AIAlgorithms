import os
import time
import tensorflow as tf
import gym
from pong_agents import Pong_RNN_PoliGrad_Agent

def rnn_main():
    # hyperparameters
    num_layer_neurons_RNN = [6400, 200, 1]

    # location of saved files (networks, speciation graphs, videos)
    directory = os.getcwd() + "/tmp/RNN"

    # parameters for monitoring and recording games
    running_reward_RNN = None
    prev_running_reward_RNN = 0
    best_reward_RNN = None
    episode_number_RNN = 0
    time_rnn, reward_rnn = [0], [0]
    record_rate_RNN = 100

    render = False

    # tensorflow session init
    sess_rnn = tf.Session()

    # openAI env init
    envRNN = gym.make('Pong-v0')

    # openAI env recording videos init
    envRNN = gym.wrappers.Monitor(envRNN,
                                 directory,
                                 force=True,
                                 video_callable=lambda episode_id: 0 == episode_number_RNN % record_rate_RNN)

    # file creation of NN
    fileRNN = open(directory + '/RNNInfo', 'w')
    fileRNN.write('episode,time,score\n')
    # fileBatchNN = open(directory + 'NN/NNBatchInfo', 'w')
    # fileBatchNN.write('batch,batch_loss,batch_entropy\n')

    # class creation of NN Agent
    rnn = Pong_RNN_PoliGrad_Agent(num_layer_neurons_RNN,
                                  sess_rnn,
                                  envRNN,
                                  directory=directory)


    while True:
        try:
            # Pong Game
            # evaluate Fully Connected Neural Network
            episode_number_RNN = rnn.episode_num
            print(episode_number_RNN)
            start = time.time()
            rnn.pong_eval()
            end = time.time()
            print("NN Reward: %i Time: %.3f" % (rnn.reward_sum, end - start))
            fileRNN.write('%i,%.3f,%i\n' % (episode_number_RNN,
                                           (end - start),
                                           rnn.reward_sum))
            fileRNN.flush()
            # fileBatchNN.write('%i,%.5f,%.8f\n' % (nn.buffer_update,
            #                                       sum(nn.batch_loss) / len(nn.batch_loss),
            #                                       sum(nn.batch_entropy) / len(nn.batch_entropy)))
            # fileBatchNN.flush()

            # score and improvement records
            running_reward_RNN = rnn.reward_sum if running_reward_RNN is None \
                else running_reward_RNN * 0.99 + rnn.reward_sum * 0.01
            if best_reward_RNN is None:
                best_reward_RNN = rnn.reward_sum
            elif rnn.reward_sum > best_reward_RNN:
                best_reward_RNN = rnn.reward_sum

            # show average score, running score, and average time
            if episode_number_RNN % rnn.poli_grad_nn.batch_size == 0:

                print(
                    'NN World Perf: Episode %f. mean reward: %f. diff: %f time: %.4f Top Score: %i' % (
                        rnn.episode_num,
                        running_reward_RNN,
                        running_reward_RNN - prev_running_reward_RNN,
                        sum(time_rnn) / len(time_rnn),
                        best_reward_RNN))
                prev_running_reward_RNN = running_reward_RNN
                time_rnn, reward_rnn = [], []
            else:
                time_rnn.append(end - start)
                reward_rnn.append(rnn.reward_sum)
            if episode_number_RNN == 1000:
                # record_rate_NN = 1000
                rnn.render_mod = 1000
            elif episode_number_RNN == 10000:
                # record_rate_NN = 2000
                rnn.render_mod = 2000
        except KeyboardInterrupt:
            fileRNN.close()

rnn_main()