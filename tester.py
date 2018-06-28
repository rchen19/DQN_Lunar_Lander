
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
import matplotlib.pyplot as plt
import pandas as pd
import json
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
font = {'size': 15}
rc('font', **font)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from plotter import *
from learner import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

# DQN tester
class DQN_tester(object):
    def __init__(self, game, ann, weights, demo):
        
        self.game = game
        self.n_states = game.observation_space.shape[0]
        self.n_actions = game.action_space.n
        self.game_a_shape = 0 if isinstance(game.action_space.sample(), int) else game.action_space.sample().shape     # to confirm the shape

        #self.test_net = ann(self.n_states, self.n_actions)
        self.test_net = ann(self.n_states, self.n_actions).to(device)
        #self.test_net.cuda()
        self.test_net.load_state_dict(torch.load(weights, map_location='cpu'))


        self.demo = demo
       
        self.episode_durations = []
        self.rewards = []
        self.steps_done = 0


    def test_action(self, s):
        self.steps_done += 1
        s = torch.FloatTensor(s).unsqueeze(0).to(device) #insert axis=0
        with torch.no_grad():
            actions_value = self.test_net.forward(s)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        #print(action)
        action = action[0] if self.game_a_shape == 0 else action.reshape(self.game_a_shape)  # return the argmax index
        return action

    def test_agent(self, param_name):


        with open(os.path.join("logs","tester_test_{}.csv".format(param_name)), "w") as f: 
            f.write("Ep,Ep_r\n")

        print('\nStart testing...')

        for trial_num in range(100):
            s = self.game.reset()
            ep_r = 0
            for t in count():

                if self.demo: self.game.render()
                a = self.test_action(s)

                # take action
                s_, r, done, info = self.game.step(a)

                ep_r += r
               
                if done or t > 1000:
                    print('Ep: ', trial_num,
                          '| Ep_r: ', round(ep_r, 2))#,
                          #'| steps_done: ', self.steps_done,
                          #'| eps_threshold: ', self.eps_threshold)
                    with open(os.path.join("logs","tester_test_{}.csv".format(param_name)), "a+") as f: 
                        f.write("{},{}\n".format(trial_num, round(ep_r, 2)))

                
                    self.episode_durations.append(t+1)
                    #if self.demo>0: plot_durations(self.episode_durations, "testing")
                    self.rewards.append(ep_r)
                    #if self.demo>0: plot_rewards(self.rewards, "testing...")
                    break
                #if t > 1000:
                #    break
                s = s_
        print('Complete')
        if self.demo: self.game.render()
        self.game.close()
        #plt.ioff()
        #if self.demo>0: plt.show()
        plot_log("tester_test_{}".format(param_name))


def test_model(env, param_name):
    demo = True
    param_list_json = os.path.join("params", "param_list.json")

    if os.path.isfile(param_list_json):

        with open(param_list_json) as f:
            param_list = json.load(f)

        params = param_list[param_name]

        ann = params["ann"]

    weights = os.path.join("weights", "weights_eval_net_{}".format(param_name))
    tester = DQN_tester(env, eval(ann), weights, demo)
    #plt.ion()
    tester.test_agent(param_name)

if __name__ == '__main__':
    #print("run from run.py")
    param_name = "param_{}".format(sys.argv[1])
    #param_name = "param_0058"


    env = LunarLander()
    test_model(env, param_name)
    #plot_result("test", "Testing")




    