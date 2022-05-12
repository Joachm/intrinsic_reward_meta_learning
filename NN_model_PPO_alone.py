from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Categorical
from itertools import count
import numpy as np
from collections import deque
from global_ import *

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

inner_step_size = 1

def make_nn():

    return NN_a_random_b()


class NN_a_random_b(nn.Module):
    def __init__(self):
        super().__init__()
        # sub network
        self.ini = True

    def forward(self, observation, reward):
        if  self.ini:
            self.ini = False
            self.net_pg = PPO(28,8)
            self.net_pg.actor_net.initial = False
        state, d = observation
        #wasted = in_a[0].astype(bool)
        #state_tensor = torch.from_numpy(state).to(device)
        state = state.astype(np.float32)[:28] # to reduce the heavy of computation
        # net a. Do not need to update parameters
        action = self.net_pg(state, reward, d)

        return action

class  Intrinsic_Reward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc_loc = nn.Linear(8, 1)

    def forward(self, in_a):
        x1 = torch.tanh(self.fc1(torch.from_numpy(in_a)))
        x2 = torch.tanh(self.fc2(x1))
        x_loc = self.fc_loc(x2)
        #x_action = x_loc[:-1]
        x_reward = x_loc
        return x_reward.numpy()


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
        old_mu,old_std = self.actor_net(states)
        pi = self.actor_net.distribution(old_mu,old_std)

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

                mu,scale = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu,scale)
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
        self.sigma = nn.Linear(16,N_A)
        self.mu = nn.Linear(16,N_A)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        self.distribution = torch.distributions.Normal
        self.initial = False

    def forward(self,s):
        if self.initial:
            self.initial = False
            nn.init.normal_(self.mu.weight,mean=0.,std=0.1)
            nn.init.normal_(self.fc2.weight,mean=0.,std=0.1)
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))

        mu = torch.tanh(self.mu(x))
        log_sigma = self.sigma(x)
        #log_sigma = torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return mu,sigma

    def choose_action(self,s):
        mu,sigma = self.forward(s)
        Pi = self.distribution(mu,sigma)
        return Pi.sample().numpy()



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




