# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:26:55 2019

@author: thoma
"""

from qagent2 import QAgent2
import gym
import matplotlib.pyplot as plt
import numpy as np

write_path = '../../data/data_long_sarsa.txt'

T=1000
nb_episodes=500

env = gym.make('MountainCar-v0')
agent = QAgent2(env,T)

lengths = -np.asarray(agent.learn(nb_episodes))

agent.generate_trajectory_file(200, write_path)

fig=plt.figure()
plt.plot(np.arange(len(lengths))[::5],np.convolve(lengths,np.ones(5,)/5,mode='same')[::5],label='train')
plt.legend()
plt.show()

trajs = agent.generate_trajectories(200)
fig=plt.figure()
plt.plot(np.arange(200),[-len(traj['states']) for traj in trajs])
plt.show()