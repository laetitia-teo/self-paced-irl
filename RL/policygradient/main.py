from reinforceagent import *
import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('MountainCar-v0')

T = 10000
N = 20
I = 20

pol = GibbsPolicy(env, T, 0.5, gamma=.998)

#pol.episode(render=True)
#print(pol.gradlog(N=2, render=True))

alphas = [.5 for i in range(I)]

thetas, grads, ls = pol.learn(I, N, alphas)
norms = [np.linalg.norm(theta) for theta in thetas]

#pol.episode(render=True)
