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
        
    