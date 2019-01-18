# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:24:31 2019

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
#from cc import CC
def RBF(state,i,c,sig):
        return(np.exp( - np.linalg.norm(state-c[i])/(2*sig[i]**2)))
    
def phi(state,action,n_a,n_features, c,sig):
    phi = np.zeros(n_features*n_a)
    temp = np.exp(-np.linalg.norm(state-c,axis=1)/(2*sig**2)) #RBF
    phi[action*n_features:(action+1)*n_features] = temp
    return(phi)
    
def Q(s,a,w,n_a,n_features,c,sig):
    temp = np.dot(phi(s,a,n_a,n_features,c,sig),w)    #linear approximation
    return temp

class QAgent2():
    
    def __init__(self,env, T, discr=100, render=True, discount = 0.99, alpha=0.1,epsilon = 0.1, n_slices = 10, reward_fun=None, add=None, add_weight=1):
    
    # The number of episodes used to evaluate the quality of a policy
        self.env=env
        self.T=T
        self.discr=discr
        self.render=render
        self.discount = discount
        self.alpha=alpha
        self.epsilon = epsilon
        self.n_slices = n_slices
        self.reward_fun=reward_fun
        self.add=add
        self.add_weight = add_weight
        # maximum number of steps of a trajectory
        self.T=T
        
        # number of grids
        
        
        self.n_a = env.action_space.n
        self.n_s = env.observation_space.low.shape[0]
        self.n_features = (self.n_slices**self.n_s)
        
        self.centers = self.generate_centers()
        self.sigs = self.generate_sig()
        
        self.w = np.random.random_sample(self.n_features*self.n_a)/10-0.5 #random initialisation
        #self.w = np.zeros(self.n_features*self.n_a) #random initialisation

    
    
    def reset(self):
        self.w = np.random.random_sample(self.n_features*self.n_a)/10-0.5
        #self.w = np.zeros(self.n_features*self.n_a) #random initialisation

    
    def generate_centers(self):
        x = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], self.n_slices)
        y= np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], self.n_slices)
        xv,yv = np.meshgrid(x,y)
        xv=xv.reshape((-1,))
        yv= yv.reshape((-1,))
        c = np.stack((xv,yv), axis=1)
        return c
    
    def generate_sig(self): #our RBF circles
        return np.ones(self.n_features)/self.n_slices
    
    def chooseAction(self,state):
        eps = np.random.rand()
        if(eps<self.epsilon):
            action=self.env.action_space.sample()
        else:
            actions = np.asarray([Q(state,a,self.w,self.n_a,self.n_features, self.centers,self.sigs) for a in range(self.n_a)]) #we can find a better way instead of computing for each actions
            actions = (np.argwhere(actions == np.max(actions)))
            actions = actions.reshape((-1,1))
            index = np.random.randint(len(actions))        
            action = actions[index][0]
            
            #action = np.argmax(np.asarray([Q(state,a,w) for a in range(n_a)]))
    
        return action
    

        
    def update_w(self,state,action,state_prim,reward):
        temp1 = reward
        temp2 = self.discount*max([np.dot(phi(state_prim,action_prim,self.n_a,self.n_features,self.centers,self.sigs),self.w) for action_prim in range(self.n_a)])-np.dot(phi(state,action,self.n_a,self.n_features,self.centers,self.sigs),self.w) 
        #print(temp1,temp2)
        TD_error =temp1+temp2
        self.w += self.alpha*TD_error*phi(state,action,self.n_a,self.n_features,self.centers,self.sigs)

    def done(self, state):
        done = (np.floor(state[0]*self.discr) >= np.floor(0.5*self.discr))
        return done
    
    def episode(self,learn=True):
        
        states = []
        actions = []
        rewards = []
        next_states = []
        state=self.env.reset()
        action = self.chooseAction(state)
        #env.render()
        j=0
        done=False
        while(j<self.T):
            if (not self.done(state)) :

                j=j+1
                state_prim, reward, done, info = self.env.step(action)
                action_prim = self.chooseAction(state_prim)
                #print(state,c)
                if(self.add == None and self.reward_fun ):
                    reward = self.reward_fun.value(state_prim, 1)
                elif(self.add ==True and self.reward_fun):
                    reward += self.add_weight * self.reward_fun.value(state_prim, 1) 
                if(learn):
                    self.update_w(state,action,state_prim,reward)
                states.append(list(state))
                actions.append(action)
                rewards.append(reward)
                next_states.append(list(state_prim))
                #env.render()
                state=state_prim
                action=action_prim

            else:
                break
        return dict(states=states, actions=actions, rewards=rewards, next_states=next_states)
    
    def learn(self, N):
        lengths = []
        for i in tqdm(range(N)):
           lengths.append(len(self.episode()['states']))
        # final episode
        return lengths
    
    def run(self): #generate trajectories
        state=self.env.reset()
        action = self.chooseAction(state,0,self.w)
        self.env.render()
        j=0
        done=False
        while(not done and j<self.T):
            state, reward, done, info = self.env.step(action)
            action = self.chooseAction(state)
            self.env.render()
            
    def generate_trajectories(self, n_traj):
        eps_temp = self.epsilon
        self.epsilon=0. #necessary to run trajectories without epsilon search
        # with or without q updates ?
        traj = []
        for i in range(n_traj):
            traj.append(self.episode(learn=False))
            
        self.epsilon = eps_temp
        return traj
    
    def generate_trajectory_file(self, n_traj, write_path):
        traj = self.generate_trajectories(n_traj)
        with open(write_path, 'w') as f:
            for t in traj:
                f.write(str(t) + '\n')
