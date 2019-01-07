import sys

sys.path.append('..')
sys.path.append('../..')

import numpy as np
import gym
import matplotlib.pyplot as plt
import readtrajectory as read
import estimatepolicy as estim
import RL.policygradient.reinforceagent
import gradientIRL as irl

env = gym.make('MountainCar-v0')
T = 1000
data_path = '../../data/data.txt'

print('hello')

# Read the data

data = read.read(data_path)
print(len(data))

# Estimate the policy parameters

policy = estim.GibbsPolicy(env, T, 1.)

trace = policy.fit(data, 500)

print(trace[-1])
print(policy.get_theta())
policy.episode(render=True)
for i in range(10):
    policy.episode()

plt.plot([t[0] for t in trace])
plt.plot([t[2] for t in trace])
plt.show()

env.close()

print('solving the IRL problem:')

dx = 5
dv = 5

reward = irl.Reward(dx, dv)
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
alphas = girl.solve()

plt.plot(alphas)
plt.show()

print(alphas)


# Solve the optimization problem






































