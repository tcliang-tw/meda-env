#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import time

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


from my_net import SimpleCnnPolicy, SimpleCnnLstmPolicy, SimpleCnnLnLstmPolicy,\
        VggCnnPolicy, VggCnnLstmPolicy, VggCnnLnLstmPolicy

from meda import*

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2, A2C
import csv

policy_names_to_policies = {'VggCnnPolicy': VggCnnPolicy,
        'VggCnnLnLstmPolicy': VggCnnLnLstmPolicy}

algorithm_names_to_algorithms = {'PPO': PPO2,
        'A2C': A2C}


def plotAgentPerformance(a_rewards, policy_name, algorithm_name, path_log, b_path = False):
    a_rewards = np.array(a_rewards).astype(np.float)
    print("Shape of A_rewards: ", a_rewards.shape)
    a_line = np.average(a_rewards, axis = 0)
    a_max = np.max(a_rewards, axis = 0)
    a_min = np.min(a_rewards, axis = 0)
    episodes = list(range(len(a_max)))
    with plt.style.context('ggplot'):
        plt.rcParams.update({'font.size': 20})
        plt.figure()
        plt.fill_between(episodes, a_max, a_min, facecolor = 'red', alpha = 0.3)
        plt.plot(episodes, a_line, 'r-', label = 'Agent')
        if b_path:
            leg = plt.legend(loc = 'upper left', shadow = True, fancybox = True)
        else:
            leg = plt.legend(loc = 'lower right', shadow = True, fancybox = True)
        leg.get_frame().set_alpha(0.5)
        plt.title(policy_name)
        plt.xlabel('Training Epochs')
        if b_path:
            plt.ylabel('Number of Cycles')
        else:
            plt.ylabel('Score')
        plt.tight_layout()
        path_fig = os.path.join(path_log, 'plot_score.png')
        plt.savefig(path_fig)

if __name__ == '__main__':

    # parameters to create the environment
    args = {'w': 30, 'l': 60,
            'n_droplets': 1,
            'b_degrade': False,
            'per_degrade': 0.1}

    algorithm_names = algorithm_names_to_algorithms.keys()
    policy_names = policy_names_to_policies.keys()

    for algorithm_name in algorithm_names:
        for policy_name in policy_names:

            # the path to where log files will be saved
            # example path: log/30_60/PPO_SimpleCnnPolicy
            path_log = os.path.join('log',
                    '_'.join( ( str(args['w']), str(args['l']) )),
                    str(args['n_droplets']),
                    '_'.join((algorithm_name, policy_name))
                    )

            agent_reward_filename = 'a_rewards.csv'
            path_agent_reward = os.path.join(path_log, agent_reward_filename)
            if not os.path.exists(path_agent_reward):
                raise Exception('The reward does not exist')

            with open(path_agent_reward, "r") as a:
                wr_a = csv.reader(a, delimiter=',')
                a_rewards = list(wr_a)
            plotAgentPerformance(a_rewards, policy_name, algorithm_name, path_log, b_path = False)

