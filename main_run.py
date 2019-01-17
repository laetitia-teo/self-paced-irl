# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 23:42:51 2019

@author: thoma
"""

import sys

sys.path.append('./IRL/GradientIRL')
sys.path.append('./IRL')
sys.path.append('./utils')

import numpy as np
import gym
import matplotlib.pyplot as plt
import readtrajectory as read
#import estimatepolicy as estim
import utils.gibbspolicy as gp
import utils.reward as rew
import gradientIRL as irl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from SelfPaced import Self_Paced
from RL.qlearning.qagent import QAgent
from RL.Sarsa.sarsa import SARSA
from tqdm import tqdm

env = gym.make('MountainCar-v0')
T = 1000
N = 1
nb_episodes=2000

data_path = 'data/data_long.txt'
write_path_girl = 'trained_models/reward_params_girl.txt'
write_path_self_paced = 'trained_models/reward_params_girl_self_paced.txt'

dx = 10
dv = 10

reward_IRL = rew.Reward(dx, dv,env)
reward_IRL.import_from_file(write_path_girl)

reward_SP = rew.Reward(dx, dv,env)
reward_SP.import_from_file(write_path_self_paced)

# =============================================================================
# qa = QAgent(env,T)
# qaIRL = QAgent(env, T, reward_fun=reward_IRL)
# qaSP = QAgent(env, T, reward_fun=reward_SP)
# qaIRLadd = QAgent(env, T, reward_fun=reward_IRL,add=True,add_weight=10.)
# qaSPadd = QAgent(env, T, reward_fun=reward_SP,add=True,add_weight=10.)
# 
# =============================================================================
qa = SARSA(env,T)
qaIRL = SARSA(env, T, reward_fun=reward_IRL)
qaSP = SARSA(env, T, reward_fun=reward_SP)
qaIRLadd = SARSA(env, T, reward_fun=reward_IRL,add=True,add_weight=200.)
qaSPadd = SARSA(env, T, reward_fun=reward_SP,add=True,add_weight=200.)

qs=[qa,qaIRL,qaSP,qaIRLadd,qaSPadd]

lengths = np.zeros(nb_episodes)
lengthsIRL = np.zeros(nb_episodes)
lengthsSP = np.zeros(nb_episodes)
lengthsIRLadd = np.zeros(nb_episodes)
lengthsSPadd = np.zeros(nb_episodes)

ls=[lengths,
lengthsIRL ,
lengthsSP ,
lengthsIRLadd ,
lengthsSPadd ,]

dict_=['Standard','IRL','Self_paced IRL','Standard+IRL','Standard+Self_paced IRL']

# =============================================================================
# qs= [SARSA(env,T)]
# ls= [np.zeros(nb_episodes)]
# dict_=['SARSA']
# =============================================================================

assert(len(ls)==len(qs))
for i in tqdm(range(N)):
    for i in range(len(qs)):
        qs[i].reset()
        ls[i]-=np.asarray(qs[i].learn(nb_episodes))
# =============================================================================
#     qa.reset()
#     qaIRL.reset()
#     qaSP.reset()
#     qaIRLadd.reset()
#     qaSPadd.reset()
#     lengths = np.asarray(qa.q_learn(nb_episodes))
#     lengthsIRL = np.asarray(qaIRL.q_learn(nb_episodes))
#     lengthsSP = np.asarray(qaSP.q_learn(nb_episodes))
#     lengthsIRLadd = np.asarray(qaIRLadd.q_learn(nb_episodes))
#     lengthsSPadd = np.asarray(qaSPadd.q_learn(nb_episodes))
# =============================================================================

# =============================================================================
# lengths = 1/N * lengths
# lengths_r = 1/N * lengths_r
# =============================================================================

#trajs = qa.generate_trajectories(50)

for i in range(len(qs)):
    plt.plot(ls[i],label='Reward: dict_[i]',linestyle = 'None',markersize=10., marker='o')
    plt.legend()
# =============================================================================
# plt.plot([length for length in lengths])
# plt.plot([length for length in lengths_r])
# =============================================================================
plt.show()

