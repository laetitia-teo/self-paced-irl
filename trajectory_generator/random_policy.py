# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:33:33 2018

@author: thoma
"""

import numpy as np
import gym

env = gym.make('MountainCar-v0')

def random_policy(state,env): 
    #action_space is dicreet
    return(np.random.randint(env.action_space.n))
    
class Random_policy():
    def __init__(self,env):
        self.actions = env.action_space
        
    def draw_action(self,state):
        return(np.random.randint(self.actions.n))
    
def collect_episodes(mdp, policy=None, horizon=None, n_episodes=1, render=False):
    paths = []
    for _ in range(n_episodes):
        observations = []
        actions = []
        rewards = []
        next_states = []

        state = mdp.reset()
        for _ in range(horizon):
            action = policy.draw_action(state)
            next_state, reward, terminal, _ = mdp.step(action)
            if render:
                mdp.render()
            observations.append(list(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(list(next_state))
            #state = copy.copy(next_state)
            if terminal:
                # Finish rollout if terminal state reached
                break
                # We need to compute the empirical return for each time step along the
                # trajectory
        paths.append(dict(
            states=observations,
            actions=actions,
            rewards=rewards,
            next_states=next_states
        ))
    return paths

if __name__=='__main__':
    horizon = 1000
    n_episodes = 15
    
    env=gym.make('MountainCar-v0')
    policy = Random_policy(env)
    paths = collect_episodes(env,policy=policy,horizon=horizon,n_episodes=n_episodes)
    
    with open('data_random_trajectories.txt', 'w') as f:
            for t in paths:
                f.write(str(t) + '\n')
    
        
# =============================================================================
#     with open('random_trajectories.pkl', 'rb') as f:
#         paths = pickle.load(f)
# =============================================================================
    

