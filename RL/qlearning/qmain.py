#import keyboard as kb
import time
from qagent import *
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
write_path = 'data_long.txt'
T = 1000

qa = QAgent(env, 1000)

#qa.episode(1, render=True)

qa.q_learn(20000)

trajs = qa.generate_trajectories(50)

plt.plot([len(t) for t in trajs])

proceed = input('Proceed to trajectory generation ?')

if proceed == 'y':
    qa.generate_trajectory_file(200, write_path)
else:
    qa.q_learn(20000)





