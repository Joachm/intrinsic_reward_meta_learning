import multiprocessing
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import cma # Covariance matrix adaptation evolution strategy (CMA-ES)
import matplotlib.pyplot as plt

import global_

from NN_model import make_nn
import my_gym as gym
#import gym

repeat_time = 200
#random_gravity = True
#random_inhibit_action = True
env_num = 4
device = torch.device('cpu')
load_task = '1'


def fitness_(worker_args):
    environment_seed, population_para = worker_args
    if pool is not None: # run paralelly
        worker_args_list = []
        for z in range(env_num):
            worker_args_list.append( (population_para) )
        rewards  = pool.map(_fitness_single_env, worker_args_list[z])
    else:
        rewards = []
        for z in range(env_num):
            worker_args = (population_para)
            rewards.append( _fitness_single_env(worker_args) )

    return rewards



def _fitness_single_env(parameters,):

    nn_model = make_nn()
    torch.nn.utils.vector_to_parameters( torch.tensor(parameters, dtype=torch.float32, device=device), nn_model.parameters() )
    #nn_model.net_pg.actor_net.initial = False
    #import global_
    #global_.set_task(load_task)
    env = gym.make('CartPole-v0')
    reward_record = []
    for i_episode in range(repeat_time):
        #try:
            reward_episode = 0
            observation = env.reset()
            done = False
            for t in range(10000):
                action = nn_model((observation,done))
                observation, reward, done, info = env.step(action)
                reward_episode += reward
                if done:
                    action = nn_model((observation,done)) # network needs update.
                    break
            reward_record.append(reward_episode)
        #except:
            #reward_episode = 0.001
            #reward_record.append(reward_episode)
    return reward_record

if __name__ == '__main__':
    def set_parallel(num_threads):
        num_threads = multiprocessing.cpu_count() if num_threads == -1 else num_threads
        num_threads = num_threads if num_threads<=12 else 12
        global pool 
        pool = multiprocessing.Pool(num_threads) if num_threads > 1 else None


    env_seed = 1
    set_parallel(env_num)


    global_.set_task(load_task)
    auto_save_file = 'result_data/'+global_.network+global_.environment+'.pkl'
    with open(auto_save_file, 'rb') as f:  # Python 3: open(..., 'rb')
        fitness_env, es = pickle.load(f)
        #para = es.result[0]
        para = es.result.xfavorite
    worker_args = env_seed, para
    reward_set = fitness_(worker_args)
    teat_auto_save_file = 'result_data/test_'+global_.network+global_.environment+'.pkl'
    with open(teat_auto_save_file, 'wb') as f:  # save the results, which can be reloaded
        pickle.dump([reward_set], f)
    fit_normal = np.mean(reward_set,axis=0)
    fit_normal_max = np.max(reward_set,axis=0)
    fit_normal_std = np.std(reward_set,axis=0)


    # plot
    x = np.arange(repeat_time)
    plt.plot(x, fit_normal, 'k', color='#1B2ACC', label='original')
    plt.fill_between(x, fit_normal-fit_normal_std, fit_normal+fit_normal_std,
        alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=0, linestyle='dashdot', antialiased=True)

    '''plt.plot(x, fit_random, 'k', color='#eb7d52', label='random weights')
    plt.fill_between(x, fit_random-fit_random_std, fit_random+fit_random_std,
        alpha=0.2, edgecolor='#eb7d52', facecolor='#ebb098',
        linewidth=0, linestyle='dashdot', antialiased=True)'''


    plt.xlabel('episode')
    plt.ylabel('fitness')

    plt.legend()
    plt.grid()
    plt.show()



