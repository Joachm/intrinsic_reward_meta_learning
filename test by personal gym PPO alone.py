import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import cma # Covariance matrix adaptation evolution strategy (CMA-ES)
import matplotlib.pyplot as plt

import global_

from NN_model_PPO_alone import make_nn
import my_gym as gym
#import gym

repeat_time = 200
#random_gravity = True
#random_inhibit_action = True
env_num = 20
device = torch.device('cpu')
import warnings
#warnings.simplefilter('error')

def fitness_(worker_args):
    environment_seed, parameters = worker_args
    reward_set = []

    def _generate_random_environment_seed(environment_seed):
        np.random.seed(environment_seed)
        envs_seed = np.random.randint(1e9, size=env_num)
        return envs_seed
    envs_seed = _generate_random_environment_seed(environment_seed)
    for i in range(len(envs_seed)):
        reward_single_env = _fitness_single_env(parameters, (envs_seed[i],i)) # a number
        reward_set.append(reward_single_env) # a list

    return reward_set



def _fitness_single_env(parameters, seed_i=None):

    nn_model = make_nn()
    #torch.nn.utils.vector_to_parameters( torch.tensor(parameters, dtype=torch.float32, device=device), nn_model.parameters() )
    nn_model.net_pg.actor_net.initial = False
    import global_
    env = gym.make(global_.environment)
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
    load_task = '1'
    env_seed = 1

    global_.set_task(load_task)
    auto_save_file = 'result_data/'+global_.network+global_.environment+'.pkl'
    
    worker_args = env_seed, 0
    reward_set = fitness_(worker_args)
    teat_auto_save_file = 'result_data/test_PPO_alone.pkl'
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



