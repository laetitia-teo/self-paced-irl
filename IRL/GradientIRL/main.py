import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../utils')

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

env = gym.make('MountainCar-v0')
T = 1000
data_path = '../../data/data_long.txt'
write_path = 'reward_params.txt'

# Read the data

data = read.read(data_path)

# Estimate the policy parameters

policy = gp.GibbsPolicy(env, T, 2.)

print('fitting policy to data...')

#trace = policy.fit(data, 200)

#print(trace[-1])
#print(policy.get_theta())

#plt.plot([t[0] for t in trace])
#plt.plot([t[2] for t in trace])
#plt.show()


policy.set_theta(np.array([-18, -1, 18]))
policy.episode(render=True)

policy.episode(render=True)
for i in range(10):
    policy.episode()

env.close()

print('solving the IRL problem:')


dx = 50
dv = 50


reward = rew.Reward(dx, dv,env)

L = dx*dv



# def plot(p):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for idx in range(L):
#         j = int(idx % dv)
#         i = int((idx - j)/dv)
#         ax.scatter(i, j, p[dv*i+j], c='r')
#     plt.show()
# 
# plot(reward.params)
# =============================================================================

'''
print('')

l = []
for i in range(N):
    r = reward.basis([1.8/N * i - 0.6, 0.0], 15, 15)
    l.append(r)

plt.plot(l)
plt.show()
'''

girl = irl.GIRL(reward, policy)
trajs = girl.import_data(data)
#girl.compute_jacobian()
#print(girl.jacobian)
alphas = girl.solve(trajs)

# =============================================================================
# plt.plot(alphas)
# =============================================================================
#plt.show()

#plot(alphas)

reward.set_params(alphas)

reward.export_to_file(write_path)
reward.import_from_file(write_path)

X = 50
V = 50



x = np.arange(-1.2, 0.6, 0.1)
v = np.arange(-0.07, 0.07, 0.005)
X = len(x)
V = len(v)
print(X)
print(V)
x, v = np.meshgrid(x, v)

r = np.zeros([X, V])

fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(X):
    for j in range(V):
        xi = i / (X-1) * 1.8 - 1.2
        vj = j / (V-1) * 0.14 - 0.07
        r[i, j] = reward.value([xi, vj], 1)
# =============================================================================
#         r[i,j] = reward.basis([xi,vj],0)
# =============================================================================
print(x.shape)
print(v.shape)
print(r.shape)
ax.plot_surface(x, v, r.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()

# =============================================================================
# girl = irl.GIRL(reward, policy)
# trajs = girl.import_data(data)
# #girl.compute_jacobian()
# #print(girl.jacobian)
# alphas = girl.solve(trajs)
# 
# #plt.plot(alphas)
# #plt.show()
# 
# #plot(alphas)
# 
# reward.set_params(alphas)
# 
# reward.export_to_file(write_path)
# #reward.import_from_file(write_path)
# 
# X = 50
# V = 50
# 
# 
# 
# x = np.arange(-1.2, 0.6, 0.1)
# v = np.arange(-0.07, 0.07, 0.005)
# X = len(x)
# V = len(v)
# print(X)
# print(V)
# x, v = np.meshgrid(x, v)
# 
# r = np.zeros([X, V])
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(X):
#     for j in range(V):
#         xi = i / (X-1) * 1.8 - 1.2
#         vj = j / (V-1) * 0.14 - 0.07
#         r[i, j] = reward.value([xi, vj], 1)
# print(x.shape)
# print(v.shape)
# print(r.shape)
# ax.plot_surface(x, v, r.T, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# 
# plt.show()
# =============================================================================

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(X):
    for j in range(V):
        xi = i / (X-1) * 1.8 - 0.6
        vj = j / (V-1) * 0.14 - 0.07
        ax.scatter(i, j, reward.value([xi, vj], 1), c='r')
plt.show()
'''









































