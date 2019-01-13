import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../utils')

import numpy as np
import gym
import matplotlib.pyplot as plt
import readtrajectory as read
import estimatepolicy as estim
import utils.gibbspolicy as gp
import gradientIRL as irl
from mpl_toolkits.mplot3d import Axes3D

env = gym.make('MountainCar-v0')
T = 1000
data_path = '../../data/data_long.txt'

# Read the data

data = read.read(data_path)

# Estimate the policy parameters

policy = gp.GibbsPolicy(env, T, 2.)

print('fitting policy to data...')

trace = policy.fit(data, 200)

#print(trace[-1])
#print(policy.get_theta())
policy.episode(render=True)
for i in range(10):
    policy.episode()

#plt.plot([t[0] for t in trace])
#plt.plot([t[2] for t in trace])
#plt.show()


#policy.set_theta(np.array([-18, -1, 18]))
#policy.episode(render=True)
env.close()

print('solving the IRL problem:')

dx = 5
dv = 5

reward = irl.Reward(dx, dv)

L = dx*dv

def plot(p):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx in range(L):
        j = int(idx % dv)
        i = int((idx - j)/dv)
        ax.scatter(i, j, p[dv*i+j], c='r')
    plt.show()

plot(reward.params)

'''
print('')

l = []
for i in range(N):
    r = reward.basis([1.8/N * i - 0.6, 0.0], 15, 15)
    l.append(r)

plt.plot(l)
plt.show()
'''

girl = irl.GIRL(reward, data, policy)
girl.compute_jacobian()
print(girl.jacobian)
girl.print_jacobian()
alphas = girl.solve()

plt.plot(alphas)
plt.show()

plot(alphas)









































