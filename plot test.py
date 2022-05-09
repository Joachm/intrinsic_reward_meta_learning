import numpy as np
import pickle
import torch
import cma # Covariance matrix adaptation evolution strategy (CMA-ES)
import matplotlib.pyplot as plt

from fitness import Fitness, set_parallel
import global_


def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))


if __name__ == '__main__':
    WSZ = 11
    repeat_time = 200

    # load and continue
    auto_save_file = 'result_data/test_NN_abCartPole-v0.pkl'
    with open(auto_save_file, 'rb') as f:  # Python 3: open(..., 'rb')
        reward_set = pickle.load(f)[0]

    fit_normal = np.mean(reward_set,axis=0)
    fit_normal_max = np.max(reward_set,axis=0)
    fit_normal_std = np.std(reward_set,axis=0)


    # plot
    x = np.arange(repeat_time)
    plt.plot(x, fit_normal, 'k', color='#1B2ACC', label='Ours')
    plt.fill_between(x, fit_normal-fit_normal_std, fit_normal+fit_normal_std,
        alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=0, linestyle='dashdot', antialiased=True)

    ###############################################
    auto_save_file = 'result_data/test_NN_a_random_bCartPole-v0.pkl'
    with open(auto_save_file, 'rb') as f:  # Python 3: open(..., 'rb')
        reward_set = pickle.load(f)[0]

    fit_normal = np.mean(reward_set,axis=0)
    fit_normal_max = np.max(reward_set,axis=0)
    fit_normal_std = np.std(reward_set,axis=0)


    plt.plot(x, fit_normal, 'k', color='#eb7d52', label='Only evolve intrinsic rewards')
    plt.fill_between(x, fit_normal-fit_normal_std, fit_normal+fit_normal_std,
        alpha=0.2, edgecolor='#eb7d52', facecolor='#ebb098',
        linewidth=0, linestyle='dashdot', antialiased=True)

    ###############################################
    auto_save_file = 'result_data/test_PPO_alone.pkl'
    with open(auto_save_file, 'rb') as f:  # Python 3: open(..., 'rb')
        reward_set = pickle.load(f)[0]

    fit_normal = np.mean(reward_set,axis=0)
    fit_normal_max = np.max(reward_set,axis=0)
    fit_normal_std = np.std(reward_set,axis=0)


    plt.plot(x, fit_normal, 'k', color='#2eb030', label='PPO alone')
    plt.fill_between(x, fit_normal-fit_normal_std, fit_normal+fit_normal_std,
        alpha=0.2, edgecolor='#2eb030', facecolor='#2eb030',
        linewidth=0, linestyle='dashdot', antialiased=True)

    ###############################################
    auto_save_file = 'result_data/test_NN_forward_hebbianCartPole-v0.pkl'
    with open(auto_save_file, 'rb') as f:  # Python 3: open(..., 'rb')
        reward_set = pickle.load(f)[0]

    fit_normal = np.mean(reward_set,axis=0)
    fit_normal_max = np.max(reward_set,axis=0)
    fit_normal_std = np.std(reward_set,axis=0)


    plt.plot(x, fit_normal, 'k', color='#e87474', label='Hebbian')
    plt.fill_between(x, fit_normal-fit_normal_std, fit_normal+fit_normal_std,
        alpha=0.2, edgecolor='#e87474', facecolor='#e87474',
        linewidth=0, linestyle='dashdot', antialiased=True)


    plt.xlabel('episode')
    plt.ylabel('fitness')

    plt.legend()
    plt.grid()
    plt.show()




