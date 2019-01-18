import keyboard as kb
import time
from testfile import *

class Session():
    
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.action = self.env.action_space.sample()
        self.done = False
        self.state_trace = []
        self.rew_trace = []
    
    def right_callback(self):
        print('right')
        state, rew, done, _ = self.env.step(2)
        self.state = state
        
        
    
    def left_callback(self):
        print('left')
        state, rew, done, _ = self.env.step(0)
        self.state = state
        self.done = done
        
    
    def update(self, state, rew):
        self.state_trace.append(state)
        self.rew_trace.append(rew)
        self.done = done

env = gym.make('MountainCar-v0')

T = 1000

# =============================

print('press esc to exit')


sess = Session(env)

kb.add_hotkey('right', sess.right_callback)
kb.add_hotkey('left', sess.left_callback)

kb.add_hotkey('space', quit)

waittime = 0.3

for _ in range(T):
    t0 = time.time()
    sess.env.render()
    if (sess.done):
        break
    #sess.right_callback()
    t = time.time() - t0
    print(time.time() - t0)
    

print(sess.state_trace)
R = 0
for r in sess.rew_trace:
    R += r

print('\nScore : {}'.format(R))
kb.unhook_all_hotkeys()

qa = QAgent(env, 1000, 10000)

#qa.episode(1, render=True)

qa.q_learn()
    
print('goodbye')
