# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:02:03 2019

@author: thoma
"""

class IRL():
        
    def zero(self):
        raise NotImplementedError
    
    def loss(self, trajs):
        '''
        Returns array of objective functions on list of trajectories
        '''
        raise NotImplementedError
    
    def loss2(self, w, trajs):
        '''
        Returns array of objective functions on list of trajectories
        '''
        raise NotImplementedError
    
    def solve(self,trajs):
        '''
        Returns solution to optimisation problem
        '''
        raise NotImplementedError
        
    def import_data(self,data):
        trajs = []
        for i in range(len(data)):
            traj = []           # building a single trajectory
            T = len(data[i]['states'])
            for t in range(T):
                state = data[i]['states'][t]
                action = data[i]['actions'][t]
                traj.append([state, action])
            trajs.append(traj)
        return trajs
        
    