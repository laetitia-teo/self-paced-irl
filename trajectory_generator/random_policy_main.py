# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:54:55 2018

@author: thoma
"""
import gym
from random_policy import *

horizon = 1000
n_episodes = 15

env=gym.make('MountainCar-v0')
policy = Random_policy(env)
paths = collect_episodes(env,policy=policy,horizon=horizon,n_episodes=n_episodes)

with open('../data/data_random_trajectories.txt', 'w') as f:
        for t in paths:
            f.write(str(t) + '\n')


