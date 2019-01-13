#from reinforceagent import *
import sys
import gym
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../utils')

import gibbspolicy as gp

env = gym.make('MountainCar-v0')

T = 1000
N = 20
I = 20

expert = gp.GibbsPolicy(env, T, 2.)
expert.set_theta(np.array([-20., 0, 20.]))

expert.episode(render=True)
'''
pol = GibbsPolicy(env, T, 0.5, gamma=.998)

#pol.episode(render=True)
#print(pol.gradlog(N=2, render=True))

alphas = [.5 for i in range(I)]

thetas, grads, ls = pol.learn(I, N, alphas)
norms = [np.linalg.norm(theta) for theta in thetas]

#pol.episode(render=True)
'''

env.close()
