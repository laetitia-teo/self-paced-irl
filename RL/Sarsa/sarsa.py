# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 01:03:27 2019

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
#from cc import CC
def RBF(state,i,c,sig):
        return(np.exp( - np.linalg.norm(state-c[i])/(2*sig[i])))
    
def phi(state,action,n_a,n_features, c,sig):
    phi = np.zeros(n_features*n_a)
    temp = np.exp(-np.linalg.norm(state-c,axis=1)/(2*sig)) #RBF
    phi[action*n_features:(action+1)*n_features] = temp
    return(phi)
    
def Q(s,a,w,n_a,n_features,c,sig):
    temp = np.dot(phi(s,a,n_a,n_features,c,sig),w)    #linear approximation
    return temp

class SARSA():
    
    def __init__(self,env, T, discr=100, render=True, discount = 0.995, alpha=0.01, gamma=1., epsilon = 0.1, n_slices = 10, reward_fun=None, add=None, add_weight=1):
    
    # The number of episodes used to evaluate the quality of a policy
        self.env=env
        self.T=T
        self.discr=discr
        self.render=render
        self.discount = discount
        self.alpha=alpha
        self.gamma=gamma
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
        
        self.w = np.random.random_sample(self.n_features*self.n_a) #random initialisation

    
    
    def reset(self):
        self.w = np.random.random_sample(self.n_features*self.n_a)
    
    def generate_centers(self):
        x = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], self.n_slices)
        y= np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], self.n_slices)
        xv,yv = np.meshgrid(x,y)
        xv=xv.reshape((-1,))
        yv= yv.reshape((-1,))
        c = np.stack((xv,yv), axis=1)
        return c
    
    def generate_sig(self): #our RBF circles
        return np.ones(self.n_features)/20
    
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
    

        
    def update_w(self,state,action,state_prim,action_prim,reward):
        TD_error =reward+self.discount*np.dot(phi(state_prim,action_prim,self.n_a,self.n_features,self.centers,self.sigs),self.w)-np.dot(phi(state,action,self.n_a,self.n_features,self.centers,self.sigs),self.w) 
        self.w += self.alpha*TD_error*phi(state,action,self.n_a,self.n_features,self.centers,self.sigs)

    def done(self, state):
        done = (np.floor(state[0]*self.discr) >= np.floor(0.5*self.discr))
        return done
    
    def episode(self,N):
        
        states = []
        actions = []
        rewards = []
        next_states = []
        state=self.env.reset()
        action = self.chooseAction(state)
        #env.render()
        j=0
        done=False
        while(not done and j<self.T):
            if not self.done(state):

                j=j+1
                state_prim, reward, done, info = self.env.step(action)
                action_prim = self.chooseAction(state_prim)
                #print(state,c)
                if(self.add == None and self.reward_fun ):
                    reward = self.reward_fun.value(state_prim, 1)
                elif(self.add ==True and self.reward_fun):
                    reward += self.add_weight * self.reward_fun.value(state_prim, 1) 
                self.update_w(state,action,state_prim,action_prim,reward)
                action=action_prim
                states.append(list(state))
                actions.append(action)
                rewards.append(reward)
                next_states.append(list(state_prim))
                #env.render()
                state=state_prim
            else:
                break
        return dict(states=states, actions=actions, rewards=rewards, next_states=next_states)
    
    def learn(self, N,plot_final=False):
        lengths = []
        for i in tqdm(range(N)):
           lengths.append(len(self.episode(N)['states']))
        # final episode
        if(plot_final):
            self.episode(0.0, render=True)
        return lengths
    
    def run(self,w): #generate trajectories
        state=self.env.reset()
        action = self.chooseAction(state,0,self.w)
        self.env.render()
        j=0
        done=False
        while(not done and j<self.T):
            state, reward, done, info = self.env.step(action)
            action = self.chooseAction(state,0,self.w)
            self.env.render()