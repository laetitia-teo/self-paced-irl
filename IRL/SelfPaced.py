# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:06:55 2018

@author: thoma
"""
import numpy as np
import scipy.optimize as opt
from IRL import IRL

#Self paced

class Self_Paced(IRL):
    
    def __init__(self,f,K0,eps,data,model=None,constraint='hard'):
        self.f = f #class created by laetitia, comme GIRL par exemple, objet IRL
        self.K = K0 
        self.eps=eps
        # params correspond to the ws in the paper.
        if self.model is None:
            self.params = self.zero()
        else:
            self.params = self.model.get_params() # this may be subject to change
        self.constraint = constraint
        #self.data=data
        self.trajs = []
        for i in range(self.N):
            traj = []           # building a single trajectory
            T = len(data[i]['states'])
            for t in range(T):
                state = data[i]['states'][t]
                action = data[i]['actions'][t]
                traj.append([state, action])
            self.trajs.append(traj)
            
        self.w = self.f.reward.params #w is the weight of our model, see GIRL example
        self.v = - np.ones(len(self.trajs)) #start
    
    def zero(self):
        self.f.zero()
        
    def reg(self,w):
        #for regularisation
        return 0
        
    def fit(self,X,Y):
        start=True #for first iteration
        v0 = np.random.rand(len(self.trajs))
        old_v = self.v
        
        ws = []
        
        loss = []
        while((self.v == np.ones(len(self.trajs))).all()): #find a termination condition perhaps double while (alternative search, and then decrement)
            
            #Alternative search strategy
            while(start == True or not((old_v == self.v).all())):
                #minimising for v
# =============================================================================
#                 result_v = opt.minimize(self.objective_v, v0, constraints=self.v_constraints)
#                 if not result_v.success:
#                     print(result_v.message)
#                     print(result_v)
#                 self.v = result_v.x
# =============================================================================
                
                #second method use dirac
                old_v=self.v
                self.v = np.where(self.f.loss(self.trajs) < 1/self.K,1,0)
                
                #minimising for w
                result_w = opt.minimize(self.objective_w, self.w)
                if not result_w.success:
                    print(result_w.message)
                    print(result_w)
                self.w = result_w.x
                self.f.reward.set_params(self.w)
            
            ws.append(self.w)
            self.K=self.mu * self.K
        
        return ws
    
    def objective_w(self,w):
        return(np.dot(self.v, self.f.loss(w,self.trajs))+self.reg(w) ) #le reste est independant de w donc pas besoin de calculer
        
    def objective_v(self,v):
        return(np.dot(v,self.f.loss(self.trajs)) - np.sum(v)/self.K) #think about a way to only calculate objective if v is 1
        
    def objective(self,w,v):
        return(self.reg(w) + v*self.f.objective(self.w) - np.sum(v)/self.K)
        
    
        
   