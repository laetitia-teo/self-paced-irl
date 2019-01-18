#import keyboard as kb
import sys

sys.path.append('../../utils')

import time
from qagent import *
import matplotlib.pyplot as plt
import reward as rd
from tqdm import tqdm

env = gym.make('CartPole-v0')
write_path = 'cartpole_data_long.txt'
T = 300
N = 1
I = 10000

qa = QAgent(env, T, discr=50)

d = qa.q_learn(I)

plt.plot(d)
plt.show()

env.close()
