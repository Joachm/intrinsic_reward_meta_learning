import time
import numpy as np
import cma # Covariance matrix adaptation evolution strategy (CMA-ES)
import pickle
import torch
from ES_classes import OpenES


from fitness import Fitness, set_parallel
import warnings
warnings.simplefilter('error')
import global_

load_and_continue = True # 'True' will load the pervious result and continue trainning.
auto_save = True # save the result
task = '1'

EvolutionStrategy =  cma.CMAEvolutionStrategy

global_.set_task(task)
if __name__ == '__main__':
    num_threads = 12 # '-1' will use all the threads that CPU have. '1' won't use multiple threads and easy to debug.
    population_size = 100

    # load and continue
    auto_save_file = 'result_data/'+global_.network+global_.environment+'.pkl'
    if load_and_continue:
        try: # try to read the data or else start new training
            with open(auto_save_file, 'rb') as f:  # Python 3: open(..., 'rb')
                fitness_env, es = pickle.load(f)
            set_parallel(num_threads)
        except:
            fitness_env = Fitness(num_threads=num_threads)
            es = EvolutionStrategy(fitness_env.parameters, 0.1, {'popsize':population_size})
    else:
        fitness_env = Fitness(num_threads=num_threads)
        es = EvolutionStrategy(fitness_env.parameters, 0.1, {'popsize':population_size})

    def print_and_save(fitness_env, es, fitness, tic_toc):
        if es.countiter % 1 ==0:
            if auto_save:
                with open(auto_save_file, 'wb') as f:  # save the results, which can be reloaded
                    pickle.dump([fitness_env, es], f)
            tic_toc[1] = time.time()
            print('max',fitness.max(),'mean',fitness.mean(),es.countiter,int(tic_toc[1]-tic_toc[0]), 'seconds')
            tic_toc[0] = time.time()

    # start training
    tic_toc = [time.time(),0]
    while True:
        population_genome = es.ask() # next generation, a 2-D array
        fitness = fitness_env.get_fitness(population_genome)
        fit_min = np.std(fitness)/(fitness-fitness.min()+1)
        es.tell(population_genome, fit_min)
        print_and_save(fitness_env, es, fitness, tic_toc)







