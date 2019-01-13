# Implementation of Path Consistency Learning
# Author : Laetitia Teodorecu
#
# Reference : Nachum et. al. 2017

import sys
import numpy as np
from tqdm import tqdm
from copy import copy

sys.path.append('../..')

import utils.gibbspolicy as gp
import utils.parametricvalue as pv

class PCL():
    
    def __init__(self, env, T, eta_pi, eta_phi, discount, d, tau, B=1000, alpha=0):
        self.env = env
        self.T = T
        self.eta_pi = eta_pi
        self.eta_phi = eta_phi
        self.discount = discount
        self.d = d
        self.tau = tau
        self.B = B
        self.alpha = alpha
        self.pi_theta = gp.GibbsPolicy(self.env, self.T, .5, gamma=self.discount)
        self.V_phi = pv.RBFValueFunction(10, 10)
        self.buff = []
    
    def done(self, state):
        pos = state[0]
        if pos >= 0.5:
            return True
        else:
            return False
    
    def rollout_traj(self, traj, idx):
        T = len(traj)
        if (T - idx) > self.d:
            return traj[idx:idx+self.d]
        else:
            return traj[idx:]
    
    def R(self, s):
        # reward of a trajectory
        r = 0
        for _, _, rew, _ in s:
            r += rew
        return r
    
    def C(self, s):
        first_state = s[0][0]
        last_state = s[-1][0]
        c = - self.V_phi.value(first_state) + self.discount**(len(s)-1) \
            *self.V_phi.value(last_state) + self.R(s) - self.tau*self.G(s)
        return c
            
    def G(self, s):
        g = 0
        for i, [state, action, _, _] in enumerate(s):
            g += self.discount**i * self.pi_theta.logproba(state, action)
        return g
    
    def grad_G(self, s):
        grad_g = 0
        for i, [state, action, _, _] in enumerate(s):
            grad_g += self.discount**i * self.pi_theta.gradlog(state, action)
        return grad_g
    
    def grad_V(self, state):
        return self.V_phi.grad(state)
    
    def gradients(self, traj):
        delta_theta = 0
        delta_phi = 0
        for idx in range(len(traj)):
            s = self.rollout_traj(traj, idx)
            C = self.C(s)
            first_state = s[0][0]
            last_state = s[-1][0]
            delta_theta += C * self.grad_G(s) # define pi_theta
            delta_phi += C * (self.grad_V(first_state)\
                - self.discount**self.d * self.grad_V(last_state))
        return delta_theta, delta_phi
    
    def episode(self, render=False):
        traj = []
        state = self.env.reset()
        for t in range(self.T):
            if not self.done(state):
                if render:
                    self.env.render()
                action = self.pi_theta.sample(state)
                next_state, reward, _, _ = self.env.step(action)
                traj.append([state, action, reward, next_state])
                state = next_state
            else:
                break
        return traj
    
    def update(self, theta, phi):
        self.pi_theta.set_theta(theta)
        self.V_phi.set_phi(phi)
    
    def reset(self):
        self.pi_theta.set_theta(self.pi_theta.zero())
        self.V_phi.set_phi(self.V_phi.zero())
    
    def learn_one(self, N, theta=None, phi=None):
        avg_length = 0
        if theta is None:
            theta = self.pi_theta.zero()
        if phi is None:
            phi = self.V_phi.zero()
        for n in tqdm(range(N)):
            traj = self.episode()
            delta_theta, delta_phi = self.gradients(traj)
            avg_length += 1/N * len(traj)
            theta += self.eta_pi*delta_theta
            phi += self.eta_phi*delta_phi
            # TODO : replay buffer
            r = self.R(traj)
            # add in buffer
            if self.buff:
                if r >= self.buff[0]['reward']:
                    self.buff.insert(0, {'traj': traj, 'reward': r})
                else:
                    self.buff.append({'traj': traj, 'reward': r})
            else: # empty buffer
                self.buff.append({'traj': traj, 'reward': r})
            # is buffer full ?
            if len(self.buff) > self.B:
                self.buff.pop(-1) # remove last element
            # replay from buffer
            traj = np.random.choice(self.buff)['traj']
            delta_theta, delta_phi = self.gradients(traj)
            theta += self.eta_pi*delta_theta
            phi += self.eta_phi*delta_phi
        return copy(theta), copy(phi), avg_length
    
    def learn(self, I, N):
        theta = self.pi_theta.get_theta()
        phi = self.V_phi.get_phi()
        trace = [[theta, phi]]
        lengths = []
        for i in tqdm(range(I)):
            t, p, length = self.learn_one(N, theta, phi)
            trace.append([t, p])
            lengths.append(length)
            self.update(t, p)
        return trace, lengths
    
    
    
                














































        
