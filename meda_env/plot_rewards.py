#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2, ACER, DQN

from meda import*
from my_net import VggCnnPolicy, DqnVggCnnPolicy
import csv

ALGOS = {'PPO': PPO2, 'ACER': ACER, 'DQN': DQN}

def get_ave_max_min (agent_rewards):
    print(agent_rewards.shape)
    agent_line = np.average(agent_rewards, axis = 0)
    agent_max = np.max(agent_rewards, axis = 0)
    agent_min = np.min(agent_rewards, axis = 0)
    return agent_line, agent_max, agent_min

def plotRewards(args, multi_agent_rewards, baseline_rewards, path_log):
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    # PPO
    r_ave, r_max, r_min = get_ave_max_min(multi_agent_rewards['PPO'])
    x_episodes = list(range(0, 0 + 5*len(r_ave), 5))
    ax.fill_between(x_episodes, r_max, r_min, alpha = 0.3)
    ax.plot(x_episodes, r_ave, marker='v', markersize=8, label = 'PPO')
    # ACER
    r_ave, r_max, r_min = get_ave_max_min(multi_agent_rewards['ACER'])
    ax.fill_between(x_episodes, r_max, r_min, alpha = 0.3)
    ax.plot(x_episodes, r_ave, marker='o', markersize=8, label = 'ACER')
    # DQN
    r_ave, r_max, r_min = get_ave_max_min(multi_agent_rewards['DQN'])
    ax.fill_between(x_episodes, r_max, r_min, alpha = 0.3)
    ax.plot(x_episodes, r_ave, marker='^', markersize=8, label = 'DQN')
    # Baseline
    baesline = np.concatenate(list(baseline_rewards.values()), axis=0)
    r_ave, r_max, r_min = get_ave_max_min(baesline)
    ax.fill_between(x_episodes, r_max, r_min, alpha = 0.3)
    ax.plot(x_episodes, r_ave, marker='x', markersize=8, label = 'Baseline')
    ax.set_xlabel('Number of Epochs', fontsize=20)
    ax.set_ylabel('Score', fontsize=20)
    leg = ax.legend(loc='lower right', shadow = True, fancybox = True, fontsize = 18)
    leg.get_frame().set_alpha(0.5)
    plt.tight_layout()
    path_fig = os.path.join(path_log, 'icml_'+args.method+'.png')
    plt.savefig(path_fig)

def read_rewards(path_log, filename):
    path_reward = os.path.join(path_log, filename)

    if not os.path.exists(path_reward):
        raise Exception('Path %s does not exist' %path_reward)

    with open(path_reward, "r") as a:
        wr_a = csv.reader(a, delimiter=',')
        a_rewards = list(wr_a)
    a_rewards = np.array(a_rewards).astype(np.float)
    return a_rewards

def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='RL training for MEDA')
    # device
    parser.add_argument('--cuda', help='CUDA Visible devices', default='0', type=str, required=False)
    # rl training
    parser.add_argument('--method', help='The method use for rl training (centralized, sharing, concurrent)',
                        type=str, default='concurrent', choices=['centralized', 'sharing', 'concurrent'])
    parser.add_argument('--n-repeat', help='Number of repeats for the experiment', type=int, default=3)
    parser.add_argument('--start-iters', help='Number of iterations the initialized model has been trained',
                        type=int, default=0)
    parser.add_argument('--stop-iters', help='Total number of iterations (including pre-train) for one repeat of the experiment',
                        type=int, default=100)
    parser.add_argument('--n-timesteps', help='Number of timesteps for each iteration',
                        type=int, default=20000)
    # env settings
    parser.add_argument('--width', help='Width of the biochip', type = int, default = 30)
    parser.add_argument('--length', help='Length of the biochip', type = int, default = 60)
    parser.add_argument('--n-agents', help='Number of agents', type = int, default = 2)
    parser.add_argument('--b-degrade', action = "store_true")
    parser.add_argument('--per-degrade', help='Percentage of degrade', type = float, default = 0.1)
    # rl evaluate
    parser.add_argument('--n-evaluate', help='Number of episodes to evaluate the model for each iteration',
                        type=int, default=20)
    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # the path to where log files will be saved
    # example path: log/30_60/PPO_SimpleCnnPolicy
    multi_results, baseline_rewards = {}, {}
    path_log_dir = os.path.join('log', args.method, str(args.width)+'_'+str(args.length), str(args.n_agents))
    for algo in ALGOS:
        path_log = os.path.join(path_log_dir, algo+'_VggCnnPolicy')
        multi_results[algo] = read_rewards(path_log, 'multi_rewards.csv')
        baseline_rewards[algo] = read_rewards(path_log, 'baseline_rewards.csv')
    plotRewards(args, multi_results, baseline_rewards, path_log_dir)

if __name__ == '__main__':
    main()
