import sys

sys.path.append('../../utils')

import numpy as np
import gym
import matplotlib.pyplot as plt
import readtrajectory as read
#import estimatepolicy as estim
import gibbspolicy as gp
import reward_general as rew
import gradientIRL_general as irl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm

env = gym.make('MountainCar-v0')
T = 1000
data_path = '../../data/data_long.txt'
write_path = '../../reward_params_1.txt'

# Read the data

data = read.read(data_path)

# Estimate the policy parameters

policy = gp.GibbsPolicy(env, T, 2.)

#trace = policy.fit(data, 200)

#print(trace[-1])
#print(policy.get_theta())

#plt.plot([t[0] for t in trace])
#plt.plot([t[2] for t in trace])
#plt.show()

policy.set_theta(np.array([-18, -1, 18]))

dx = 10


reward = rew.Reward(dx, env)

girl = irl.GIRL(reward, policy)
trajs = girl.import_data(data)
alphas = girl.solve(trajs)
reward.set_params(alphas)

reward.plot()

reward.export_to_file(write_path)
#plot_reward(reward, 'GIRL')









































