import numpy as np
import torch
import multiprocessing


from NN_model import make_nn
#import my_gym as gym
import gym

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

env_num = 1


def set_parallel(num_threads):
    num_threads = multiprocessing.cpu_count() if num_threads == -1 else num_threads
    global pool 
    pool = multiprocessing.Pool(num_threads) if num_threads > 1 else None


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

    return np.mean(reward_set)



def _fitness_single_env(parameters, seed_i=None):
    import global_
    repeat_time = global_.repeat_time
    in_count = global_.in_count
    update_timestep = global_.update_timestep

    nn_model = make_nn()
    torch.nn.utils.vector_to_parameters( torch.tensor(parameters, dtype=torch.float32, device=device), nn_model.parameters() )
    import global_
    env = gym.make(global_.environment)
    reward_record = []
    reward_record_for_an_update = []

    episode = 0
    episode_last_update = 0
    steps = 0
    while episode < 100:
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

            while steps >= update_timestep and episode_last_update != episode: # Do at least one episode
                reward_record_for_an_update = np.append(reward_record_for_an_update,np.mean(reward_record[episode_last_update:]))
                episode_last_update = episode
                steps = 0

        reward_record = np.append(reward_record,reward_episode)
        if len(reward_record_for_an_update) >= repeat_time:
            d = reward_record_for_an_update[-in_count:] - reward_record_for_an_update[-in_count*2:-in_count]
            if np.sum(d) <= 0:
                break
        episode = episode+1
    return reward_record_for_an_update[-in_count:].mean()




class Fitness():
    def __init__(self, num_threads=1, initial_para = None):
        set_parallel(num_threads)
        self.initial_para = initial_para
        # initialize network parameters
        nn_model = make_nn()
        parameters = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().numpy()
        self.para_num = parameters.size
        self.sigma0 = parameters.std()
        self.parameters = parameters

        # record
        self.reward_record = []
        pass

    def get_fitness(self, population_para):
        if self.initial_para is not None:
            complete_para = []
            for z in range(len(population_para)):
                complete_para.append(np.append(population_para[z], self.initial_para))
            population_para = complete_para
        self.rewards = self._get_rewards(population_para)
        self.reward_record.append(self.rewards)
        return self.rewards

    def _get_rewards(self, population_para):
        environment_seed = np.random.randint(1e9)
        if pool is not None: # run paralelly
            worker_args_list = []
            for z in range(len(population_para)):
                worker_args_list.append( (environment_seed, population_para[z]) )
            rewards  = pool.map(fitness_, worker_args_list)
        else:
            rewards = []
            for z in range(len(population_para)):
                worker_args = (environment_seed, population_para[z])
                rewards.append( fitness_(worker_args) )

        rewards = np.array(rewards).astype(np.float32)
        return rewards



if __name__ == '__main__':
    main_class = Fitness(num_threads=2)

    if False:
        id_ = str(1640874105)
        file_name = 'result_data/' + id_ + '/auto_save_parameters.npy'
        with open(file_name, 'rb') as f:
            main_class.parameters = np.load(f)
            main_class.reward_test = np.load(f)
            main_class.id_ = id_
        #main_class.run(iterations = 10000)
    main_class.run(iterations = 10000, print_step=1, auto_save =True)

    #main_class.test_result()







