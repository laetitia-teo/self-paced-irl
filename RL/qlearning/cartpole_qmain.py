#import keyboard as kb
import sys

sys.path.append('../../utils')

import numpy as np
import time
from qagent import *
import matplotlib.pyplot as plt
import reward_general as rd
from tqdm import tqdm

env = gym.make('CartPole-v0')
write_path = 'cartpole_data_long.txt'
import_path = '../../data/reward_params_cartpole.txt'
T = 300
N = 500
I = 500

reward = rd.Reward(5, env)
reward.import_from_file(import_path)

qa = QAgent(env, T, discr=20)
qa_r = QAgent(env, T, discr=20, reward_fun=reward)

d = np.array(qa.q_learn(I))
d_r = np.array(qa_r.q_learn(I))

for i in range(N-1):
    qa.reset()
    qa_r.reset()
    d += np.array(qa.q_learn(I))
    d_r += np.array(qa_r.q_learn(I))



plt.plot(d/N)
plt.plot(d_r/N)
plt.show()

env.close()
