from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Categorical
from itertools import count
import numpy as np
from numba import njit
from collections import deque
from global_ import *

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

inner_step_size = 1

def make_nn():
    import global_
    nn_dict = {
        'NN_ab': NN_ab,
        'NN_a_random_b':NN_a_random_b,
        }
    return nn_dict[global_.network]()


class NN_ab(nn.Module):
    def __init__(self):
        super().__init__()
        # sub network
        self.net_ir =  Intrinsic_Reward()
        self.net_pg = PPO(4,1)

    def forward(self, observation):
        state, d = observation
        #wasted = in_a[0].astype(bool)
        #state_tensor = torch.from_numpy(state).to(device)
        state = state.astype(np.float32)
        # net a. Do not need to update parameters
        with torch.no_grad():
            intrinsic_reward = self.net_ir(state)
        # net b
        action = self.net_pg(state, [1], d)

        return action


class NN_a_random_b(nn.Module):
    def __init__(self):
        super().__init__()
        # sub network
        self.net_ir =  Intrinsic_Reward()
        self.net_pg = PPO(4,1)

    def forward(self, observation):
        state, d = observation
        #wasted = in_a[0].astype(bool)
        #state_tensor = torch.from_numpy(state).to(device)
        state = state.astype(np.float32)
        # net a. Do not need to update parameters
        with torch.no_grad():
            intrinsic_reward = self.net_ir(state)
        # net b
        action = self.net_pg(state, intrinsic_reward, d)

        return action

class  Intrinsic_Reward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc_loc = nn.Linear(8, 1)

    def forward(self, in_a):
        x1 = torch.tanh(self.fc1(torch.from_numpy(in_a)))
        x2 = torch.tanh(self.fc2(x1))
        x_loc = self.fc_loc(x2)
        #x_action = x_loc[:-1]
        x_reward = x_loc
        return x_reward.numpy()



'''if self.initial:
    self.initial = False
    self.random_vetor = torch.rand(16)
    #self.random_vetor = torch.pow(10, self.random_vetor*2-1)-2 # range [0.1, 10)
    self.random_vetor = self.random_vetor*4-2 # range [-2, 2)
    ones = torch.ones(self.into_last_layer)
    self.random_vetor = torch.where(torch.rand(self.into_last_layer)<0.5,self.random_vetor,ones)'''
'''if self.initial:
    self.initial = False
    para = torch.nn.utils.parameters_to_vector( self.parameters() )
    mask_random = torch.randn(para.shape)*1+1
    para_random = mask_random*para
    final_para_b = torch.where(torch.rand(para.shape)<0.5,para_random,para)
    torch.nn.utils.vector_to_parameters( final_para_b, self.parameters() )'''
'''if self.initial:
    self.initial = False
    para_b = torch.nn.utils.parameters_to_vector( self.parameters() )
    para_random = torch.zeros(para_b.shape)
    mask_random = torch.rand(para_b.shape)
    mask = mask_random<0.5 
    final_para_b = torch.where(mask,para_b,para_random)
    torch.nn.utils.vector_to_parameters( final_para_b, self.parameters() )'''

class Nomalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,),dtype=np.float32)
        self.std = np.zeros((N_S, ),dtype=np.float32)
        self.stdd = np.zeros((N_S, ),dtype=np.float32)
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            #更新样本均值和方差
            old_mean = self.mean.copy()
            self.mean = old_mean + ((x - old_mean) / self.n).astype(np.float32)
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
            #状态归一化
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean

        x = x - self.mean

        x = x / (self.std + 1e-8)

        x = np.clip(x, -5, +5)
        return x


class PPO(nn.Module):
    def __init__(self,N_S,N_A):
        super().__init__()
        self.actor_net =Actor(N_S,N_A)
        self.critic_net = Critic(N_S)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(),lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(),lr=lr_critic,weight_decay=l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()
        self.memory = [[],[],[],[]] # deque()#
        self.previous_state = None
        self.steps = 0
        self.nomalize = Nomalize(N_S)


    def forward(self,state, intrinsic_reward, done):
        state = self.nomalize(state)
        d = float(done)
        mask = 1-d
        if self.previous_state is not None:
            #self.memory.append([self.previous_state.numpy(),action.numpy(),intrinsic_reward,mask])
            self.memory[0].append(self.previous_state)
            self.memory[1].append(self.previous_action)
            self.memory[2].append(intrinsic_reward)
            self.memory[3].append(mask)
        with torch.no_grad():
            action = self.actor_net.choose_action(torch.from_numpy(state))
        self.previous_action = action
        self.previous_state = state
        self.steps += 1
        if self.steps == update_timestep:
            self.train()
            self.memory = [[],[],[],[]] # deque()#
            self.previous_state = None # After the update, the state and action need to be resampled
            self.steps = 0
        return action

    def train(self):
        states = torch.tensor(np.vstack(self.memory[0]),dtype=torch.float32)
        actions = torch.tensor(np.vstack(self.memory[1]),dtype=torch.float32)
        rewards = torch.tensor(np.array(self.memory[2]),dtype=torch.float32)
        masks = torch.tensor(np.array(self.memory[3]),dtype=torch.float32)

        values = self.critic_net(states)

        returns,advants = self.get_gae(rewards,masks,values)
        old_mu = self.actor_net(states)
        pi = self.actor_net.distribution(old_mu)

        old_log_prob = pi.log_prob(actions).sum(1,keepdim=True)

        n = len(states)
        arr = np.arange(n)
        for epoch in range(20):
            np.random.shuffle(arr)
            for i in range(n//batch_size):
                b_index = arr[batch_size*i:batch_size*(i+1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index]

                mu = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu)
                new_prob = pi.log_prob(b_actions).sum(1,keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                #KL散度正则项
               # KL_penalty = self.kl_divergence(old_mu[b_index][b_index],mu)
                ratio = torch.exp(new_prob-old_prob)

                surrogate_loss = ratio*b_advants
                values = self.critic_net(b_states)

                critic_loss = self.critic_loss_func(values,b_returns)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio,1.0-epsilon,1.0+epsilon)

                clipped_loss =ratio*b_advants

                actor_loss = -torch.min(surrogate_loss,clipped_loss).mean()
                #actor_loss = -(surrogate_loss-beta*KL_penalty).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()

                self.actor_optim.step()
    #计算KL散度
    def kl_divergence(self,old_mu,old_sigma,mu,sigma):

        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()

        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / \
             (2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    #计算GAE
    def get_gae(self,rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            #计算A_t并进行加权求和
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - \
                              values.data[t]
            running_advants = running_tderror + gamma * lambd * \
                              running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        #advants的归一化
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants



class Actor(nn.Module):
    def __init__(self,N_S,N_A):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(N_S,32)
        self.fc2 = nn.Linear(32,16)
        self.mu = nn.Linear(16,N_A)
        self.distribution = torch.distributions.Bernoulli
        self.initial = True; self.random_vetor = None
        self.into_last_layer = 16
    #初始化网络参数
    def set_init(self,layers):
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
        if self.initial:
            self.initial = False
            self.random_vetor = torch.rand(16)
            #self.random_vetor = torch.pow(10, self.random_vetor*2-1)-2 # range [0.1, 10)
            self.random_vetor = self.random_vetor*4-2 # range [-2, 2)
            ones = torch.ones(self.into_last_layer)
            self.random_vetor = torch.where(torch.rand(self.into_last_layer)<0.5,self.random_vetor,ones)
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        if self.random_vetor is not None:
            x = x*self.random_vetor

        mu = torch.sigmoid(self.mu(x))
        return mu

    def choose_action(self,s):
        mu = self.forward(s)
        Pi = self.distribution(mu)
        return Pi.sample().numpy().astype(int)[0]

#Critic网洛
class Critic(nn.Module):
    def __init__(self,N_S):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(N_S,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self,layers):
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values




