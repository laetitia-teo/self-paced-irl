import gym
import sys

sys.path.append('../../utils')
sys.path.append('../../data')

import numpy as np
import gym
import matplotlib.pyplot as plt
import readtrajectory as read
#import estimatepolicy as estim
import gibbspolicy as gp
import reward_general as rew
import qualityfunction as qf
import gradientIRL_general as irl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm

env = gym.make('CartPole-v0')
T = 300
data_path = '../../data/data_cartpole.txt'
write_path = 'reward_params_cartpole.txt'

data = read.read(data_path)

q = qf.CartPoleQuality()
policy = gp.GibbsPolicy(env, T, 2., Q=q)

trace = policy.fit(data, 200)

reward = rew.Reward(5, env)

girl = irl.GIRL(reward, policy)
trajs = girl.import_data(data)
alphas = girl.solve(trajs)
reward.set_params(alphas)

#reward.plot()

reward.export_to_file(write_path)
