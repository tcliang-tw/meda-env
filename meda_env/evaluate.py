#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import argparse

import time
import tensorflow as tf

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2, ACER, DQN

from meda import*
from my_net import VggCnnPolicy, DqnVggCnnPolicy
import csv

ALGOS = {'PPO': PPO2, 'ACER': ACER, 'DQN': DQN}

def showIsGPU():
    if tf.test.is_gpu_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")

def EvaluateAgent(env, obs, agent, isSingleAgent = False):
    episode_reward = 0.0
    done, state = False, None
    step = 0
    while not done:
        if isSingleAgent:
            start_index = 0
            action_droplets = []
            for i in range(env.routing_manager.n_droplets):
                obs_single = np.zeros(shape = (env.width, env.length, 3))
                for j in range(env.routing_manager.n_droplets):
                    if j == i:
                        continue
                    o_drp = env.routing_manager.droplets[j]
                    obs_single = env._addDropletInObsLayer(obs_single, o_drp, 3 * 0)
                # Add destination in 3 x i + 1 layer
                dst = env.routing_manager.destinations[i]
                obs_single = env._addDropletInObsLayer(obs_single, dst, 3 * 0 + 1)
                # Add droplet in 3 x i + 2 layer
                drp = env.routing_manager.droplets[i]
                obs_single = env._addDropletInObsLayer(obs_single, drp, 3 * 0 + 2)
                action_droplet, _ = agent.predict(obs_single)
                action_droplets.append(action_droplet)
                start_index += 3
            action = 0
            for i in range(len(action_droplets)):
                action += action_droplets[i] *  pow(len(action_droplets), i)
        else:
            action, state = agent.predict(obs)
        obs, reward, done, _info = env.step(action)
        if not isSingleAgent:
            print(done)
        step+=1
        episode_reward += reward
    print(step)

    return episode_reward

def evaluateOnce(args, path_log, env, repeat_num):
    algo = ALGOS[args.algo]
    len_results = (args.stop_iters - args.start_iters)//5 + 1
    results = {'baseline': [0]*len_results, 'single': [0]*len_results, 'multi': [0]*len_results}
    for i in range(len_results):
        print('### Evaluating iteration %d' %(i*5))
        model_name = '_'.join(['repeat', str(repeat_num), 'training', str(i*5), str(args.n_timesteps)])
        path_multi = os.path.join(path_log, model_name)
        path_single = path_multi.replace(os.path.sep+str(args.num_agents)+os.path.sep,
                                         os.path.sep+'1'+os.path.sep)
        print(path_multi)
        print(path_single)
        multi_agent = algo.load(path_multi)
        single_agent = algo.load(path_single)
        baseline_agent = BaseLineRouter(env.width, env.length)
        for j in range(args.n_evaluate):
            print('### Episode %d.'%j)
            obs = env.reset()
            routing_manager = env.routing_manager
            results['baseline'][i] += baseline_agent.getEstimatedReward(routing_manager)[0]
            results['single'][i] +=  EvaluateAgent(env, obs, single_agent, isSingleAgent = True)
            results['multi'][i] +=  EvaluateAgent(env, obs, multi_agent, isSingleAgent = False)
        results['baseline'][i] /= args.n_evaluate
        results['single'][i] /= args.n_evaluate
        results['multi'][i] /= args.n_evaluate
    return results

def save_evaluation(agent_rewards, filename, path_log):
    with open(os.path.join(path_log, filename), 'w') as agent_log:
        writer_agent = csv.writer(agent_log)
        writer_agent.writerows(agent_rewards)

def evaluateSeveralTimes(args=None, path_log=None):
    showIsGPU()
    multi_rewards, single_rewards, baseline_rewards = [], [], []
    for repeat in range(1, args.n_repeat+1):
        print("### In repeat %d" %(repeat))
        start_time = time.time()
        env = MEDAEnv(w=args.width, l=args.length, n_droplets=args.num_agents,
                      b_degrade=args.b_degrade, per_degrade = args.per_degrade)
        results = evaluateOnce(args, path_log, env, repeat_num=repeat)
        print("### Repeat %s costs %s seconds ###" %(str(repeat), time.time() - start_time))
        multi_rewards.append(results['multi'])
        single_rewards.append(results['single'])
        baseline_rewards.append(results['baseline'])

    save_evaluation(multi_rewards, 'multi_rewards.csv', path_log)
    save_evaluation(single_rewards, 'single_rewards.csv', path_log)
    save_evaluation(baseline_rewards, 'baseline_rewards.csv', path_log)

def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='RL training for MEDA')
    # device
    parser.add_argument('--cuda', help='CUDA Visible devices', default='0', type=str, required=False)
    parser.add_argument('--algo', help='RL Algorithm', default='PPO', type=str, required=False, choices=list(ALGOS.keys()))
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
    parser.add_argument('--num-agents', help='Number of agents', type = int, default = 2)
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
    path_log = os.path.join('log', str(args.width)+'_'+str(args.length),
            str(args.num_agents), args.algo+'_VggCnnPolicy')
    print('### Start evaluating algorithm %s'%(args.algo))
    evaluateSeveralTimes(args, path_log = path_log)

if __name__ == '__main__':
    main()
