import numpy as numpy
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
font = {'size': 15}
rc('font', **font)

def plot_rewards(rewards, title, mean=100):
    plt.figure(3)
    plt.clf()
    rewards_t = torch.cuda.tensor(rewards, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= mean:
        means = rewards_t.unfold(0, mean, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(mean-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    #if is_ipython:
    #    display.clear_output(wait=True)
    #    display.display(plt.gcf())

def plot_durations(episode_durations, title, mean=100):
    plt.figure(2)
    plt.clf()
    durations_t = torch.cuda.tensor(episode_durations, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= mean:
        means = durations_t.unfold(0, mean, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(mean-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    #if is_ipython:
    #    display.clear_output(wait=True)
    #    display.display(plt.gcf())

def plot_result(name, title):
    infile = os.path.join("logs","{}.csv".format(name))
    outfile = os.path.join("logs","{}.png".format(name))
    result = pd.read_csv(infile, header=0, index_col=0)
    result["Rolling Mean"] = result["Ep_r"].rolling(window=100).mean()
    result.fillna(0, inplace=True)
    ax = result.plot()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    #plt.show()
    plt.savefig(outfile)

def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = getattr(getattr(plotting, '_style'), '_get_standard_colors')(num_colors=len(cols))

    # First axis
    
    if len(cols) == 2:
        ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
        ax.set_ylabel(ylabel=cols[0])
        ax.text(0.99, 0.1, s="Average: {}".format(round(data.loc[:, cols[1]].mean(), 3)), 
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=ax.transAxes)

        lines, labels = ax.get_legend_handles_labels()

    elif len(cols) == 3:
        ax = data.loc[:, [cols[0],cols[2]]].plot(label=[cols[0],cols[2]], color=[colors[0],colors[2]], **kwargs)
        ax.set_ylabel(ylabel=cols[0])
        lines, labels = ax.get_legend_handles_labels()

        for n in [1]:
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
            data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)])
            ax_new.set_ylabel(ylabel=cols[n])

            # Proper legend position
            line, label = ax_new.get_legend_handles_labels()
            lines += line
            labels += label

    ax.legend(lines, labels, loc=0)
    return ax

def plot_log(fname):
    infile = os.path.join("logs", "{}.csv".format(fname))
    outfile = os.path.join("logs", "{}.png".format(fname))

    result = pd.read_csv(infile, header=0, index_col=0)
    result.index.names = ['Episode']
    if len(result.columns) == 2:
        result.columns = ['Reward', 'Epsilon']
    elif len(result.columns) == 1:
        result.columns = ['Reward']
    result["Rolling Mean"] = result["Reward"].rolling(window=100).mean()
    #result.fillna(0, inplace=True)
    #print(result)

    plot_multi(result, figsize=(10, 5))
    plt.savefig(outfile)
    plt.close()


if __name__ == '__main__':
    print("run from run.py")
