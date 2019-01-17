# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 02:52:37 2019

@author: thoma
"""

from sarsa import SARSA
import gym
import matplotlib.pyplot as plt
import numpy as np

write_path = '../../data/data_long_sarsa.txt'

T=1000
nb_episodes=500

env = gym.make('MountainCar-v0')
agent = SARSA(env,T)

lengths = -np.asarray(agent.learn(nb_episodes))
agent.generate_trajectory_file(200, write_path)

plt.plot(np.arange(len(lengths))[::5],np.convolve(lengths,np.ones(5,)/5,mode='same')[::5])
plt.show()

