import numpy as np
import pickle
import torch
import cma # Covariance matrix adaptation evolution strategy (CMA-ES)
import matplotlib.pyplot as plt

from fitness import Fitness, set_parallel
import global_

load_task = '3'

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

    # load and continue
    global_.set_task(load_task)
    auto_save_file = 'result_data/'+global_.network+global_.environment+'150.pkl'
    with open(auto_save_file, 'rb') as f:  # Python 3: open(..., 'rb')
        species, es = pickle.load(f)


    WSZ = 1
    rr = np.array(species.reward_record)
    r_max = smooth(np.max(species.reward_record, axis=1),WSZ)
    r_mean = smooth(np.mean(species.reward_record, axis=1),WSZ)
    r_median = smooth(np.median(species.reward_record, axis=1),WSZ)
    r_std = smooth(np.std(species.reward_record, axis=1),WSZ)



    x = np.arange(r_max.size)
    plt.plot(x, r_max, '-k', label='max')
    plt.plot(x, r_mean, 'k', color='#1B2ACC', label='mean')
    plt.fill_between(x, r_mean-r_std, r_mean+r_std,
        alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=0, linestyle='dashdot', antialiased=True)


    plt.legend()
    plt.grid()
    plt.show()
    pass






