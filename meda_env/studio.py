#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import time
from PIL import ImageDraw
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


from my_net import VggCnnPolicy, VggCnnLstmPolicy, VggCnnLnLstmPolicy

from meda import*

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2, A2C
import csv


policy_names_to_policies = {'VggCnnPolicy': VggCnnPolicy,
        'VggCnnLnLstmPolicy': VggCnnLnLstmPolicy}

algorithm_names_to_algorithms = {'PPO': PPO2}


def getGameLength(env):
    ans = env.agt_pos[0] - env.agt_end[0]
    ans += env.agt_pos[1] - env.agt_end[1]
    return ans

def drawGridInImg(image, step):
    draw = ImageDraw.Draw(image)
    y_top = 0
    y_btm = image.height
    for x in range(0, image.width, step):
        line = ((x, y_top), (x, y_btm))
        draw.line(line, fill = "rgb(64, 64, 64)")
    x_lef = 0
    x_rig = image.width
    for y in range(0, image.height, step):
        line = ((x_lef, y), (x_rig, y))
        draw.line(line, fill = "rgb(64, 64, 64)")
    return image

def runAGame(model, env):
    obs = env.reset()
    #env.envs[0].b_random = True
    #repeat = random.randrange(0, 100)
    #for i in range(repeat):
    #    env.reset()
    #    obs = env.reset()
    images = []
    done, state = False, None
    while not done:
        action, state = model.predict(obs)
        obs, reward, done, _info = env.step(action)
        done = done[0]
        img = env.envs[0].render('rgb_array')
        frame = Image.fromarray(img)
        scale = int(200 / env.envs[0].width)
        frame = frame.resize((env.envs[0].length * scale,
                env.envs[0].width * scale))
        frame = drawGridInImg(frame, scale)
        images.append(frame)
    #env.envs[0].b_random = False
    return images[:-1]

def recordVideo(env, model, filename):
    """ Record videos for three games """
    # env = model.get_env()
    images = []
    images = images + runAGame(model, env)
    images = images + runAGame(model, env)
    images = images + runAGame(model, env)
    images[0].save(filename + '.gif',
            format='GIF',
            append_images=images[1:],
            save_all=True,
            duration=500,
            loop=0)
    print('Video saved:', filename)


def trainNRecordASetting(env, path_log, algorithm_name, policy_name,
        n_iterations, n_timesteps, repeat_num = 1):
    algorithm = algorithm_names_to_algorithms[algorithm_name]
    video_name = os.path.join(path_log, 'images')
    Path(video_name).mkdir(parents=True, exist_ok=True)
    # No modules first
    for i in range(0, n_iterations + 1):
        if i % 5 == 0:
            model_name = '_'.join(('repeat', str(repeat_num),\
                                'training', str(i), str(n_timesteps)))
            multi_agent = algorithm.load(os.path.join(path_log,
                                            model_name))
            recordVideo(env, multi_agent, os.path.join(video_name, str(i)))

def recordTrainingProcess(args, path_log,
                        algorithm_name, policy_name,
                        n_iterations = 300, n_timesteps = 20000, repeat_num = 1):
    env = make_vec_env(
            MEDAEnv, n_envs = 4, env_kwargs = args)
    trainNRecordASetting(env, path_log,
        algorithm_name, policy_name,
        n_iterations=n_iterations, n_timesteps = 20000,
        repeat_num = 1)
    return

if __name__ == '__main__':

    # parameters to create the environment
    args = {'w': 80, 'l': 60,
            'n_droplets': 2,
            'b_degrade': False,
            'per_degrade': 0.1}
    # parameters to in the training process
    repeat_n = 1 # Which repeat to use for gif creation
    num_iteration = 300 # number of iteration runned in one repeat

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

            recordTrainingProcess(args, path_log,
                    algorithm_name, policy_name,
                    n_iterations = num_iteration, n_timesteps = 20000, repeat_num = repeat_n)
            print('### Finished studio.py successfully ###')
