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
from SelfPaced import Self_Paced

env = gym.make('MountainCar-v0')
T = 1000
data_path = '../../data/data_long.txt'
write_path_girl = 'reward_params_girl.txt'
write_path_self_paced = 'reward_params_girl_self_paced.txt'

def plot_reward(reward,title):
    X = 50
    V = 50
    
    sp =reward.env.observation_space
    
    x = np.linspace(sp.low[0], sp.high[0], X)
    v = np.linspace(sp.low[1], sp.high[1],V)

    x, v = np.meshgrid(x, v)
    
    r = np.zeros([X, V])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(X):
        for j in range(V):
            xi = i / (X-1) * (sp.high[0] - sp.low[0]) + sp.low[0]
            vj = j / (V-1) * (sp.high[1] - sp.low[1]) + sp.low[1]
            r[i, j] = reward.value([xi, vj], 1)
    # =============================================================================
    #         r[i,j] = reward.basis([xi,vj],0)
    # =============================================================================
    ax.plot_surface(x, v, r.T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_title(title)
    
    plt.show()

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

for i in range(10):
    policy.episode()

env.close()

print('solving the IRL problem:')

dx = 10
dv = 10


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

# plt.plot(alphas)
# =============================================================================
#plt.show()

#plot(alphas)

reward.set_params(alphas)
# 
reward.export_to_file(write_path_girl)

reward.import_from_file(write_path_girl)

plot_reward(reward,'GIRL algorithm')

reward_sp = rew.Reward(dx, dv,env)
f_sp = irl.GIRL(reward_sp, policy)
K0=10e4
eps=10e-15 #not working for now
mu=0.5  

girl_self_paced = Self_Paced(f_sp,K0,eps,mu)
trajs = girl_self_paced.import_data(data)
# =============================================================================
# alphass = girl_self_paced.fit1(trajs)
# =============================================================================
alphass = girl_self_paced.fit2(trajs)

#plt.plot(alphas)
#plt.show()

#plot(alphas)

print(len(alphass))

reward_sp.set_params(alphass[-1])

reward_sp.export_to_file(write_path_self_paced)
#reward.import_from_file(write_path)

for i in range(len(alphass)):
    reward_sp.set_params(alphass[i])

    plot_reward(reward_sp,'Self-Paced GIRL algorithm : iteration %d' % (i))
    

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









































