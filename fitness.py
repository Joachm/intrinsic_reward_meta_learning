import numpy as np
import torch
import multiprocessing


from NN_model import make_nn
#import my_gym as gym
import gym
import pybullet
import pybullet_envs

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

env_num = 1


def set_parallel(num_threads):
    num_threads = multiprocessing.cpu_count() if num_threads == -1 else num_threads
    global pool 
    pool = multiprocessing.Pool(num_threads) if num_threads > 1 else None


def fitness_(worker_args):
    parameters, random_vector = worker_args
    reward_set = []

    for i in range(env_num):
        reward_single_env = _fitness_single_env(parameters, random_vector) # a number
        reward_set.append(reward_single_env) # a list

    return np.mean(reward_set)



def _fitness_single_env(parameters, random_vector=None):
    import global_
    repeat_time = global_.repeat_time
    in_count = global_.in_count
    inner_loop_max_steps = global_.inner_loop_max_steps

    nn_model = make_nn()
    torch.nn.utils.vector_to_parameters( torch.tensor(parameters, dtype=torch.float32, device=device), nn_model.parameters() )
    import global_
    env = gym.make(global_.environment)
    env._max_episode_steps = 500 # from 1000 to 500 to reduce the heavy.
    reward_record = []

    steps = 0 # training
    while steps < inner_loop_max_steps:
        observation = env.reset()
        reward_episode = 0
        done = False
        while not done:
            steps += 1
            action = nn_model((observation,done))
            observation, reward, done, info = env.step(action)
            reward_episode += reward
            if done: # network needs record.
                _ = nn_model((observation,done))

            '''while steps >= update_timestep and episode_last_update != episode: # Do at least one episode
                reward_record_for_an_update = np.append(reward_record_for_an_update,np.mean(reward_record[episode_last_update:]))
                episode_last_update = episode
                steps = 0'''
        reward_record = np.append(reward_record,reward_episode)

    reward_record_after_update = []
    for i in range(3): # test the performance of agent in inner loop
        observation = env.reset()
        reward_episode = 0
        done = False
        while not done:
            steps += 1
            action = nn_model((observation,done))
            observation, reward, done, info = env.step(action)
            reward_episode += reward
            if done: # network needs record.
                _ = nn_model((observation,done))
        reward_record_after_update = np.append(reward_record_after_update,reward_episode)
    return reward_record_after_update.mean()





class Fitness():
    def __init__(self, num_threads=1, initial_para = None):
        set_parallel(num_threads)
        #self.initial_para = initial_para
        # initialize network parameters
        nn_model = make_nn()
        parameters = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().numpy()
        self.para_num = parameters.size
        self.sigma0 = parameters.std()
        self.parameters = parameters

        # for early stopping
        self.generation = 0
        self.not_early_stopping = True
        self.random_shift = np.exp(np.random.randn(48,28))
        self.validation_record = []

        # record
        self.reward_record = []
        pass
    '''
    def _get_initial_parameters(self):
        parameters = np.load('pre_train_intrinsic_reward.npy')
        return parameters
    '''

    def _get_initial_parameters(self):
        nn_model = make_nn()
        parameters = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().numpy()
        return parameters

    def get_fitness(self, population_para):
        self.generation += 1
        '''if self.initial_para is not None:
            complete_para = []
            for z in range(len(population_para)):
                complete_para.append(np.append(population_para[z], self.initial_para))
            population_para = complete_para'''

        rewards = self._get_rewards(population_para)
        self.reward_record.append(rewards)
        return rewards

    def _get_rewards(self, population_para):
        if pool is not None: # run paralelly
            worker_args_list = []
            for z in range(len(population_para)):
                worker_args_list.append( (population_para[z], None) )
            rewards  = pool.map(fitness_, worker_args_list)
        else:
            rewards = []
            for z in range(len(population_para)):
                worker_args = (population_para[z], None)
                rewards.append( fitness_(worker_args) )

        rewards = np.array(rewards).astype(np.float32)
        return rewards


    def check_validation(self,best_para):
        in_count = 5
        if self.generation % 5 == 0: # for each x generation
            validation_rewards = self._get_validation_rewards(best_para.astype(np.float32), self.random_shift)

            self.validation_record = np.append(self.validation_record,np.mean(validation_rewards))
            if len(self.validation_record) >=  2* in_count:
                d = self.validation_record[-in_count:] - self.validation_record[-in_count*2:-in_count]
                if np.sum(d) <= 0:
                    self.not_early_stopping = False


    def _get_validation_rewards(self, best_para, random_vector):
        if pool is not None: # run paralelly
            worker_args_list = []
            for z in range(len(random_vector)):
                worker_args_list.append( (best_para, random_vector[z]) )
            rewards  = pool.map(fitness_for_validation, worker_args_list)
        else:
            rewards = []
            for z in range(len(random_vector)):
                worker_args = (best_para, random_vector[z])
                rewards.append( fitness_for_validation(worker_args) )

        #rewards = np.array(rewards).astype(np.float32)
        return rewards




def fitness_for_validation(worker_args):
    parameters, random_vector = worker_args
    import global_
    in_count = global_.in_count

    nn_model = make_nn()
    torch.nn.utils.vector_to_parameters( torch.tensor(parameters, dtype=torch.float32, device=device), nn_model.parameters() )
    import global_
    env = gym.make(global_.environment)
    env._max_episode_steps = 500 # from 1000 to 500 to reduce the heavy.
    reward_record = []

    episode = 0
    episode_last_update = 0
    steps = 0
    while episode < 50:
        observation = env.reset()
        reward_episode = 0
        done = False
        while not done:
            steps += 1
            action = nn_model((observation*random_vector,done))
            observation, reward, done, info = env.step(action)
            reward_episode += reward
            if done: # network needs record.
                _ = nn_model((observation*random_vector,done))

        reward_record = np.append(reward_record,reward_episode)
        episode = episode+1
    return reward_record[-30:].mean()






