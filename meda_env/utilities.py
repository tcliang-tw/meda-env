#!/usr/bin/python
"""
Trainers that use baseline algorithms for the multi-agent envrionment
2020/12/20 Tung-Che Liang
"""

import os
import gym
import threading
from meda import*
from my_net import VggCnnPolicy, VggCnnLnLstmPolicy
from stable_baselines import PPO2, ACER, DQN

class LearnThread(threading.Thread):
    def __init__(self, model, total_timesteps):
        super(LearnThread, self).__init__()
        self.model = model
        self.total_timesteps = total_timesteps

    def run(self):
        print("### Running thread", self.model.env.envs[0].agent_index, "...")
        self.model.learn(self.total_timesteps)
        print("### Finished thread", self.model.env.envs[0].agent_index)

class DecentrailizedTrainer:
    """
    This trainer uses baseline models under the hood for pettingzoo envs
    """
    def __init__(self, policy, parallel_env, model_type, concurrent = True):
        assert(isinstance(parallel_env, ParallelEnv))
        if model_type not in ['PPO', 'ACER', 'DQN']:
            raise TypeError("{} is not a legal model type".format(model_type))
        self.models = {}
        self.p_env = parallel_env
        self.concurrent = concurrent
        if concurrent:
            for agent in parallel_env.agents:
                self.models[agent] = self.getModel(
                        model_type,
                        copy.deepcopy(policy),
                        ConcurrentAgentEnv(parallel_env, agent))
        else: # parameter sharing
            model = self.getModel(
                    model_type,
                    policy,
                    ParaSharingEnv(parallel_env))
            for agent in parallel_env.agents:
                self.models[agent] = model

    def getModel(self, model_type, policy, env):
        if model_type == "PPO":
            return PPO2(policy, env)
        elif model_type == "ACER":
            return ACER(policy, env)
        else: # DQN
            return DQN(policy, env)

    def learn(self, total_timesteps):
        if self.concurrent:
            threads = []
            for agent in self.p_env.agents:
                model = self.models[agent]
                thd = LearnThread(model, total_timesteps)
                thd.start()
                threads.append(thd)
            for thd in threads:
                thd.join()
        else:
            self.models[self.p_env.agents[0]].learn(total_timesteps)

    def save(self, save_path):
        if self.concurrent:
            for i, agent in enumerate(self.p_env.agents):
                model = self.models[agent]
                model.save(save_path + '_c{}'.format(i))
        else:
            self.models[self.p_env.agents[0]].save(save_path + 'shared')

class ConcurrentAgentEnv(gym.Env):
    """ Single Agent for MEDAEnv(ParallelEnv) """
    metaddata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, env, agent):
        super(ConcurrentAgentEnv, self).__init__()
        self.env = env
        if agent not in self.env.agents:
            raise TypeError("{} is not one of the agents in {}".format(
                    agent, env))
        self.agent = agent
        self.count = 0
        self.agent_index = self.env.agents.index(agent)
        self.action_space = self.env.action_spaces[agent]
        self.observation_space = self.env.observation_spaces[agent]
        self.reward_range = (-1.0, 1.0)

    def step(self, action):
        self.count += 1
        reward = self.env.routing_manager.moveOneDroplet(
                self.agent_index, action, self.env.m_health, True)
        obs = self.env.getOneObs(self.agent_index)
        if self.count <= self.env.max_step:
            done = self.env.routing_manager.getTaskStatus()[self.agent_index]
        else:
            done = True
        return obs, reward, done, {}

    def reset(self):
        #print('### Resetting task', self.agent_index, '...')
        self.count = 0
        self.env.routing_manager.resetTask(self.agent_index)
        return self.env.getOneObs(self.agent_index)

    def render(self, mode = 'human'):
        return self.env.render(mode)

    def close(self):
        pass

class ParaSharingEnv(gym.Env):
    """ Single Agent for MEDAEnv(ParallelEnv) """
    metaddata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, env):
        super(ParaSharingEnv, self).__init__()
        self.env = env
        self.agent_index = 0
        self.count = 0
        agent = env.agents[0]
        self.action_space = self.env.action_spaces[agent]
        self.observation_space = self.env.observation_spaces[agent]
        self.reward_range = (-1.0, 1.0)

    def step(self, action):
        reward = self.env.routing_manager.moveOneDroplet(
                self.agent_index, action, self.env.m_health)
        if self.agent_index == len(self.env.agents) - 1:
            self.agent_index = 0
            self.count += 1
        else:
            self.agent_index += 1
        obs = self.env.getOneObs(self.agent_index)
        if self.count > self.env.max_step:
            done = True
        else:
            done = self.env.routing_manager.getTaskStatus()[self.agent_index]
        return obs, reward, done, {}

    def reset(self):
        self.env.routing_manager.refresh()
        self.agent_index = 0
        self.count = 0
        return self.env.getOneObs(self.agent_index)

    def render(self, mode = 'human'):
        return self.env.render(mode)

    def close(self):
        pass

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    p_env = MEDAEnv(30, 60, 4)
    model = DecentrailizedTrainer(VggCnnPolicy, p_env, 'PPO', True)
    model.learn(1000)

    model = DecentrailizedTrainer(VggCnnPolicy, p_env, 'ACER', False)
    model.learn(100)
    print("Finished utilities.py")
