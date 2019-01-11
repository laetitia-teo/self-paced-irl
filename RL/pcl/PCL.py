# Implementation of Path Consistency Learning
# Author : Laetitia Teodorecu
#
# Reference : Nachum et. al. 2017

import sys
import numpy as np

sys.path.append('../..')

import utils.gibbspolicy as gp
import parametricvalue as pv

class PCL():
    
    def __init__(self, env, T, eta_pi, eta_psi, discount, d, tau, B=0, alpha=0):
        self.env = env
        self.T = T
        self.eta_pi = eta_pi
        self.eta_psi = eta_psi
        self.discount = discount
        self.d = d
        self.tau = tau
        self.B = N
        self.alpha = alpha
        self.pi_theta = gp.GibbsPolicy(self.env, self.T, .5, gamma=self.discount)
        self.V_psi = pv.ParametricValueFunction()
    
    def done(self, state):
        pos = state[0]
        if pos >= 0.5:
            return True
        else:
            return False
    
    def rollout_traj(self, traj, idx):
        T = len(traj)
        if (T - idx) > d:
            return traj[idx:idx+d]
        else:
            return traj[idx:]
    
    def R(self, s):
        r = 0
        for _, _, rew, _ in s:
            r += rew
        return r
    
    def C(self, s):
        c = - self.V_psi(s[0]) + self.discount**(len(s)-1)*self.V_psi(s[-1])
            + self.R(s) - self.tau*self.G(s)
    
    def G(self, s):
        g = 0
        for i, [state, action, _, _] in enumerate(s):
            g += gamma**i * self.pi_theta.logproba(action, state)
    
    def grad_G(self, s)
        grad_g = 0
        for i, [state, action, _, _] in enumerate(s):
            grad_g += gamma**i * self.pi_theta.gradlog(action, state)
    
    def grad_V(self, state):
        return self.V_psi.grad(state)
    
    def gradients(self, traj):
        delta_theta = 0
        delta_psi = 0
        for idx in len(traj):
            s = self.rollout_traj(traj, idx)
            C = self.C(s)
            delta_theta += C * grad_G(s) # define pi_theta
            delta_psi += C * (grad_V(s[0]) - self.discount**d * grad_V(s[-1]))
        return delta_theta, delta_psi
    
    def episode(self, render=False):
        traj = []
        state = self.env.reset()
        for t in range(self.T):
            if not self.done(state):
                if render:
                    self.env.render()
                action = self.sample(state)
                next_state, reward, _, _ = self.env.step(action)
                traj.append([state, action, reward, next_state])
                state = next_state
            else:
                break
        return traj
    
    def learn(self, N):
        for n in tqdm(range(N)):
            theta = self.pi_theta.zero()
            psi = self.V_psi.zero()
            traj = self.episode()
            delta_theta, delta_phi = self.gradients(traj)
            theta += self.eta_pi*delta_theta
            psi += self.eta_psi*delta_psi
            # TODO : replay buffer
    
    
    
                














































        
