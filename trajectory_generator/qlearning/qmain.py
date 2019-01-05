#import keyboard as kb
import time
from qagent import *
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

T = 1000

qa = QAgent(env, 1000)

#qa.episode(1, render=True)

qa.q_learn(20000)

trajs = qa.generate_trajectories(15)
pos = []

print(trajs)

plt.figure()

for t in trajs:
    l = []
    for s in t['states']:
        l.append(s[0])
    pos.append(l)

for p in pos:
    plt.plot(p)

plt.show()

qa.generate_trajectory_file(15)



