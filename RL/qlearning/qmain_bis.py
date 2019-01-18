# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 03:31:29 2019

@author: thoma
"""

from qagent import *
import gym
import matplotlib.pyplot as plt
import numpy as np

write_path = '../../data/data_long_sarsa.txt'

T=1000
nb_episodes=1000

env = gym.make('MountainCar-v0')
agent = QAgent(env,T)

lengths = -np.asarray(agent.q_learn(nb_episodes))
# =============================================================================
# agent.generate_trajectory_file(5000, write_path)
# =============================================================================

plt.plot(np.arange(len(lengths))[::5],np.convolve(lengths,np.ones(5,)/5,mode='same')[::5])
plt.show()
