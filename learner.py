
import math
import sys
import os
import random
from itertools import count
from collections import namedtuple
import numpy as np
import gym
from gym.envs.box2d.lunar_lander import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


from itertools import product
import os
import json
from plotter import *
from tester import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# make a neural net to approximate Q function
class Net_01(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net_01, self).__init__()
        self.fc1 = nn.Linear(n_states, 80)
        #self.fc1.weight.data.normal_(0, 0.0001)   # initialization
        self.out = nn.Linear(80, n_actions)
        #self.out.weight.data.normal_(0, 0.0001)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        #x = F.sigmoid(x)
        actions_value = self.out(x)
        return actions_value


# a larger net
class Net_02(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net_02, self).__init__()
        self.fc1 = nn.Linear(n_states, 80)
        #self.fc1.weight.data.normal_(0, 0.0001)   # initialization
        self.fc2 = nn.Linear(80, 80)
        #self.out.weight.data.normal_(0, 0.0001)   # initialization
        self.out = nn.Linear(80, n_actions)
        #self.out.weight.data.normal_(0, 0.0001)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        #x = F.sigmoid(x)
        actions_value = self.out(x)
        return actions_value

# a even larger net
class Net_03(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net_03, self).__init__()
        self.fc1 = nn.Linear(n_states, 80)
        #self.fc1.weight.data.normal_(0, 0.0001)   # initialization
        self.fc2 = nn.Linear(80, 80)
        #self.out.weight.data.normal_(0, 0.0001)   # initialization
        self.fc3 = nn.Linear(80, 80)
        #self.out.weight.data.normal_(0, 0.0001)   # initialization
        self.out = nn.Linear(80, n_actions)
        #self.out.weight.data.normal_(0, 0.0001)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        #x = F.sigmoid(x)
        actions_value = self.out(x)
        return actions_value


# an experience memory class
class ExpMemory(object):

    def __init__(self, memory_capacity, n_states, batch_size):
        assert memory_capacity > batch_size, "batch size larger than capacity"
        self.capacity = memory_capacity
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.capacity, n_states * 2 + 3))     # initialize memory
        self.batch_size = batch_size

    def sample(self, ):
        assert self.memory_counter >= self.batch_size, "not enough experience"
        sample_index = np.random.choice(min(self.capacity, self.memory_counter), self.batch_size, replace=True)
        return self.memory[sample_index, :]

    def __len__(self):
        return len(self.memory)

    def store_transition(self, s, a, r, s_next, d):
        transition = np.hstack((s, [a, r], s_next, d))
        # replace the old memory with new memory
        index = self.memory_counter % self.capacity
        self.memory[index, :] = transition
        self.memory_counter += 1



# DQN learner
class DQN(object):
    def __init__(self, game, ann, loss, optim,\
                gamma, memory_capacity, \
                lr, lr_decay, \
                eps_end, eps_start, eps_decay, \
                target_replace_iter, batch_size, max_episode, \
                param_name, demo,\
                tau=None, dd=False, learn_interval=5,\
                decay_schedule=1, eps_step_limit=2500,
                converge_episode=100, converge_value=220):
        
        self.game = game
        self.n_states = game.observation_space.shape[0]
        self.n_actions = game.action_space.n
        self.game_a_shape = 0 if isinstance(game.action_space.sample(), int) else game.action_space.sample().shape     # to confirm the shape

        self.eval_net = ann(self.n_states, self.n_actions).to(device)
        self.target_net = ann(self.n_states, self.n_actions).to(device)
         self.test_net = ann(self.n_states, self.n_actions).to(device)
        self.max_episode = max_episode
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.decay_schedule = decay_schedule
        self.eps_decay1 = eps_decay
        self.eps_decay2 = (self.eps_end / self.eps_start) ** (1/self.max_episode)
        self.lr = lr
        self.lr_decay = lr_decay
        self.eps_threshold = eps_start
        self.learn_step_counter = 0                                  # for target updating
        self.target_replace_iter = target_replace_iter
        self.gamma = gamma
        
        self.param_name = param_name
        self.demo = demo
        self.tau = tau
        self.dd = dd
        self.learn_interval = learn_interval
        self.batch_size = batch_size
        self.eps_step_limit = eps_step_limit
        self.converge_episode = converge_episode
        self.converge_value = converge_value

        
        
        self.episode_durations = []
        self.rewards = []
        self.memory = ExpMemory(memory_capacity, self.n_states, batch_size)
        

        if optim == "adam": self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        if optim == "rms": self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=lr)
        if loss == "mse": self.loss_func = nn.MSELoss()
        if loss == "huber": self.loss_func = F.smooth_l1_loss # Huber loss
        self.steps_done = 0

    def choose_action(self, s, eps_threshold):
        #sample = random.random()
        
        self.steps_done += 1
        s = torch.FloatTensor(s).unsqueeze(0).to(device) #insert axis=0
        # input only one sample
        if np.random.uniform() >= eps_threshold:   # greedy
            with torch.no_grad():
                actions_value = self.eval_net.forward(s)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0] if self.game_a_shape == 0 else action.reshape(self.game_a_shape)  # return the argmax index
        else:   # random
            #action = np.random.randint(0, self.n_actions)
            #action = action if self.game_a_shape == 0 else action.reshape(self.game_a_shape)
            action = self.game.action_space.sample()
        return action


    def learn(self, ): #this function optimize the neural net one step according to the loss function and chosen optimizer when called
        
        self.learn_step_counter += 1

        # sample batch transitions        
        b_memory = self.memory.sample()

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(b_memory[:, -1].reshape(-1,1), device=device, dtype=torch.float32) #FloatTensor#.transpose_(0, 1)
        #print (non_final_mask.size())
        

        b_s = torch.tensor(b_memory[:, :self.n_states], device=device, dtype=torch.float32) #FloatTensor
        b_a = torch.tensor(b_memory[:, self.n_states:self.n_states+1].astype(int), device=device, dtype=torch.int64) #LongTensor
        b_r = torch.tensor(b_memory[:, self.n_states+1:self.n_states+2], device=device, dtype=torch.float32) #FloatTensor
        b_s_next = torch.tensor(b_memory[:, -self.n_states-1:-1], device=device, dtype=torch.float32) #FloatTensor

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_next).detach()     # detach from graph, don't backpropagate
        if self.dd: #double DQN

            q_eval4next = self.eval_net(b_s_next).detach()
            max_act4next = q_eval4next.argmax(1)
            q_next = q_next[:, max_act4next][:,0].view(self.memory.batch_size, 1)
            q_next = non_final_mask * q_next
        else: #normal DQN
            q_next = non_final_mask * q_next.max(1, keepdim=True)[0]

        q_target = b_r + self.gamma * q_next   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        #this part defines how target net is updated
        if self.tau is not None: #soft update
            
            params_eval = self.eval_net.named_parameters()
            params_target = self.target_net.named_parameters()

            dict_params_target = dict(params_target)

            for name_eval, param_eval in params_eval:
                if name_eval in dict_params_target:
                    dict_params_target[name_eval].data.copy_( self.tau * param_eval.data + (1-self.tau) * dict_params_target[name_eval].data )

            self.target_net.load_state_dict(dict_params_target)

        elif self.learn_step_counter % self.target_replace_iter == 0: #hard update
            self.target_net.load_state_dict(self.eval_net.state_dict())
            pass


    def train(self, ):

        with open(os.path.join("logs","train_{}.csv".format(self.param_name)), "w") as f: 
            f.write("Ep,Ep_r,epsilon\n")

        print('\nCollecting experience...')
        learning_starting_episode = 0
        ep_r_list = []
        for i_episode in range(self.max_episode):
            s = self.game.reset()
            ep_r = 0

            for t in count(): #starting an episode, t is the number of steps taken

                if self.demo>1: self.game.render()
                a = self.choose_action(s, self.eps_threshold)

                # take action
                s_prime, r, done, info = self.game.step(a)
                self.memory.store_transition(s, a, r, s_prime, float(not done))
                ep_r += r
                if self.memory.memory_counter > self.memory.batch_size * 1.5 and self.steps_done % self.learn_interval == 0: #perform learning every 5 steps
                    if learning_starting_episode == 0: learning_starting_episode = i_episode
                    self.learn()

                if done or t > self.eps_step_limit:
                    print(self.param_name,
                        'Ep: ', i_episode,
                        '| r: ', round(ep_r, 2),
                        '| steps: ', self.steps_done,
                          #'| mem_count: ', self.memory.memory_counter,
                          #'| batch_size: ', self.memory.batch_size,
                        '| eps: ', round(self.eps_threshold,3))
                    with open(os.path.join("logs","train_{}.csv".format(self.param_name)), "a+") as f: 
                        f.write("{},{},{}\n".format(i_episode, round(ep_r, 6), round(self.eps_threshold, 6)))

                    self.episode_durations.append(t+1)
                    if self.demo>0: plot_durations(self.episode_durations, "training")
                    self.rewards.append(ep_r)
                    if self.demo>0: plot_rewards(self.rewards, "training...")
                    break
                s = s_prime

            ep_r_list.append(ep_r)
            if len(ep_r_list) == self.converge_episode: ep_r_list.pop(0)
            if np.mean(ep_r_list) > self.converge_value: break
            if self.decay_schedule == 1 and self.memory.memory_counter > self.memory.batch_size * 1.5:
                self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                    np.exp(-1. * (i_episode-learning_starting_episode) / self.eps_decay1)
            elif self.decay_schedule == 2 and self.memory.memory_counter > self.memory.batch_size * 1.5:
                self.eps_threshold *= self.eps_decay2
            elif self.decay_schedule == 3 and self.memory.memory_counter > self.memory.batch_size * 1.5:
                if i_episode <= 200:
                    self.eps_threshold = 0.99
                elif i_episode <= 500:
                    self.eps_threshold = 0.9
                elif i_episode <= 1000:
                    self.eps_threshold = 0.8
                elif i_episode <= 2000:
                    self.eps_threshold = 0.7
                elif i_episode <= 3000:
                    self.eps_threshold = 0.6
                elif i_episode <= 4000:
                    self.eps_threshold = 0.5
                elif i_episode <= 4500:
                    self.eps_threshold = 0.3
                else:
                    self.eps_threshold = 0.1

            #save temporary model weights every 100 episodes and plot intermediate results
            if i_episode % 100 == 0:
                self.save_weights(weight_name="weights_temp", model_name="model_temp")
                plot_log("train_{}".format(self.param_name))
                
        #save model weights
        self.save_weights(weight_name="weights", model_name="model")

        print('Complete')
        if self.demo>1: self.game.render()
        self.game.close()
        #plt.ioff()
        if self.demo>0: plt.show()

    def save_weights(self, weight_name="weights", model_name="model"):
        torch.save(self.eval_net.state_dict(), os.path.join("weights","{}_eval_net_{}".format(weight_name, self.param_name)))
        torch.save(self.target_net.state_dict(), os.path.join("weights","{}_target_net_{}".format(weight_name, self.param_name)))
        torch.save(self.eval_net, os.path.join("weights","{}_eval_net_{}".format(model_name, self.param_name)))
        torch.save(self.target_net, os.path.join("weights","{}_target_net_{}".format(model_name, self.param_name)))

    def test_agent(self, ):

        with open(os.path.join("logs","test_{}.csv".format(self.param_name)), "w") as f: 
            f.write("Ep,Ep_r\n")

        print('\nStart testing...')

        for trial_num in range(100):
            s = self.game.reset()
            ep_r = 0
            for t in count():

                if self.demo>1: self.game.render()
                a = self.choose_action(s, eps_threshold=0) #choose action according to policy

                # take action
                s_prime, r, done, info = self.game.step(a)

                ep_r += r
               
                if done or t > 1000:
                    print(self.param_name, 
                        'Ep: ', trial_num,
                        '| Ep_r: ', round(ep_r, 2))#,
                          #'| steps_done: ', self.steps_done,
                          #'| eps_threshold: ', self.eps_threshold)
                    with open(os.path.join("logs","test_{}.csv".format(self.param_name)), "a+") as f: 
                        f.write("{},{}\n".format(trial_num, round(ep_r, 2)))

                
                    self.episode_durations.append(t+1)
                    if self.demo>0: plot_durations(self.episode_durations, "testing")
                    self.rewards.append(ep_r)
                    if self.demo>0: plot_rewards(self.rewards, "testing...")
                    break

                s = s_prime
        print('Complete')
        if self.demo>1: self.game.render()
        self.game.close()
        #plt.ioff()
        if self.demo>0: plt.show()

        plot_log("test_{}".format(self.param_name))


def train_model(env, params, param_name):
    demo = 0


    # Hyper Parameters
    loss = params["loss"]
    optim = params["optim"]
    max_episode = params["max_episode"]
    batch_size = params["batch_size"]
    lr = params["lr"]                # learning rate
    lr_decay = params["lr_decay"]
    tau = params["tau"] #soft update coefficient

    eps_start = params["eps_start"]
    eps_end = params["eps_end"]
    eps_decay = params["eps_decay"]
    gamma = params["gamma"]             # reward discount
    target_replace_iter = params["target_replace_iter"]  # target update frequency
    memory_capacity = params["memory_capacity"]
    dd = params["dd"]
    ann = params["ann"]
    learn_interval = params["learn_interval"]
    decay_schedule = params["decay_schedule"]
    eps_step_limit = params["eps_step_limit"]
    converge_episode = params["converge_episode"]
    converge_value = params["converge_value"]
 
    
    learner = DQN(env, eval(ann), loss, optim,\
                gamma, memory_capacity, \
                lr, lr_decay, \
                eps_end, eps_start, eps_decay, \
                target_replace_iter, batch_size, max_episode, \
                param_name, demo,\
                tau, dd,
                learn_interval, decay_schedule, eps_step_limit)
    #plt.ion()
    learner.train()
    learner.test_agent()


    pass






if __name__ == '__main__':

    print("run from run.py")

    



