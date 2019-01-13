import numpy as np
import gym
import PCL as pcl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = gym.make('MountainCar-v0')
T = 1500
eta_pi = .1
eta_phi = .001
discount = 1
d = 10
tau = 1.

discr = 10

agent = pcl.PCL(env, T, eta_pi, eta_phi, discount, d, tau)
#t, p = agent.learn_one(10)

def plot(p):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(discr):
        for j in range(discr):
            ax.scatter(i, j, p[discr*i+j], c='r')

    plt.show()

#print(t)
#plot(p)
trace, length = agent.learn(12, 10)
traces = []
lengths = []
'''
for i in range(20):
    agent.reset()
    trace, length = agent.learn(10, 10)
    traces.append(trace)
    lengths.append(length)
'''
