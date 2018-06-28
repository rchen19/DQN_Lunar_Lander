
from learner import *
from itertools import product
import os
import json
from plotter import *
from tester import *
import numpy as np
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    """
    max_episode_range = [10000]
    batch_size_range = [64]
    lr_range = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]#0.00001                   # learning rate
    lr_decay_range = [0.8]
    tau_range = [0.1] #soft update coefficient

    eps_start_range = [0.99]
    eps_end_range = [0.05]
    eps_decay_range = [1000000]#80000000
    gamma_range = [0.85, 0.90, 0.95, 0.99]#0.99              # reward discount
    target_replace_iter_range = [10]#100   # target update frequency
    memory_capacity_range = [4000]#4000#10000
    dd_range = [true, false]
    """

    max_episode_range = [2500]
    batch_size_range = [8000]
    lr_range = [5e-5]#0.00001                   # learning rate
    lr_decay_range = [0.8]
    tau_range = [0.02]#, 0.01, 0.1, 0.005] #soft update coefficient None --> hard update

    eps_start_range = [1.0]
    eps_end_range = [0.005]
    eps_decay_range = [170]#80000000 #used only when decay_schedule = 1
    gamma_range = [0.99]#, 0.95, 0.999]#0.99              # reward discount
    target_replace_iter_range = [500]#100   # target update frequency, used only when tau is None
    memory_capacity_range = [100000]#4000#10000
    dd_range = [False]
    ann_range = ["Net_02"]
    learn_interval_range = [1]
    decay_schedule_range = [1] #three different epsilon decay schedule: 1. exp(-i/400) 2. ^i, 3.pre-determined
    loss_range = ["mse"] #huber or mse
    optim_range = ["rms"] #rms or adam
    eps_step_limit_range = [2000] #upper limit for how many steps to move in one episode
    converge_episode_range = [150]
    converge_value_range = [220]

    param_list_json = os.path.join("params", "param_list.json")

    if os.path.isfile(param_list_json):

        with open(param_list_json) as f:
            param_list = json.load(f)

    else:
        param_list = dict()

    for loss, optim, max_episode, batch_size, lr, lr_decay, tau, eps_start, eps_end, eps_decay, gamma, target_replace_iter, \
        memory_capacity, dd, ann, learn_interval, decay_schedule, eps_step_limit, converge_episode, converge_value \
        in [\
            ["mse", "rms", 2500, 8000, 5e-5, 0.8, 0.02, 1.0, 0.005, 170, 0.99, 500, 100000, False, "Net_03", 1, 1, 2000, 150, 220], \
            ["mse", "rms", 2500, 8000, 5e-5, 0.8, 0.02, 1.0, 0.005, 170, 0.99, 500, 100000, False, "Net_02", 1, 1, 2000, 150, 220], \
            ["mse", "rms", 2500, 8000, 5e-5, 0.8, 0.02, 1.0, 0.005, 170, 0.99, 500, 100000, True, "Net_01", 1, 1, 2000, 150, 220], \
            ["mse", "rms", 2500, 8000, 5e-5, 0.8, 0.02, 1.0, 0.005, 170, 0.9, 500, 100000, False, "Net_03", 1, 1, 2000, 150, 220], \
            ["mse", "rms", 2500, 8000, 5e-5, 0.8, 0.02, 1.0, 0.005, 170, 0.999, 500, 100000, False, "Net_03", 1, 1, 2000, 150, 220],\
            ["huber", "rms", 2500, 8000, 5e-5, 0.8, 0.02, 1.0, 0.005, 170, 0.99, 500, 100000, False, "Net_03", 1, 1, 2000, 150, 220],\
            ["mse", "adam", 2500, 8000, 5e-5, 0.8, 0.02, 1.0, 0.005, 170, 0.99, 500, 100000, False, "Net_03", 1, 1, 2000, 150, 220],\
            ]:

        params = {"loss":loss,
                "optim":optim,
                "max_episode":max_episode, 
                "batch_size":batch_size, 
                "lr":lr, 
                "lr_decay":lr_decay, 
                "tau":tau, 
                "eps_start":eps_start, 
                "eps_end":eps_end, 
                "eps_decay":eps_decay, 
                "gamma":gamma, 
                "target_replace_iter":target_replace_iter, 
                "memory_capacity":memory_capacity, 
                "dd":dd,
                "ann": ann,
                "learn_interval":learn_interval,
                "decay_schedule":decay_schedule,
                "eps_step_limit":eps_step_limit,
                "converge_episode":converge_episode,
                "converge_value":converge_value,
                "note":""}
        param_list_keys = list(param_list.keys())
        if len(param_list_keys) == 0:
            param_name = "param_0000"
        else:
            last_number = int(sorted(param_list_keys)[-1].split("_")[-1])

            param_name = "param_" + "{:04d}".format(last_number + 1)

        param_list[param_name] = params
        data = json.dumps(param_list, indent=2)

        with open(param_list_json,"w") as f:
            f.write(data)

        df = pd.DataFrame(param_list).T
        df.to_csv(os.path.join("params", "param_list.csv"))


        #if sys.argv[1] == "carpole":
        #    env = env = gym.make('CartPole-v0').unwrapped
        #elif sys.argv[1] == "lunar":
            #env = LunarLander()
            #env = gym.make('LunarLander-v2').unwrapped

        env = LunarLander()


        train_model(env, params, param_name)
        #plot_result("train", "Training")

        #weights = sys.argv[3]
        #test_model(env)
        #plot_result("test", "Testing")
    #data = json.dumps(param_list, indent=2)

    #with open(param_list_json,"w") as f:
    #    f.write(data)

    #df = pd.DataFrame(param_list).T
    #df.to_csv(os.path.join("params", "param_list.csv"))




