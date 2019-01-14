# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:06:55 2018

@author: thoma
"""
import numpy as np
import scipy.optimize as opt
from IRL import IRL
from tqdm import tqdm
from copy import copy

#Self paced

class Self_Paced(IRL):
    
    def __init__(self,f,K0,eps,mu,model=None,constraint='hard',eps1=10e-3):
        self.f = f #class created by laetitia, comme GIRL par exemple, objet IRL
        self.K = K0 
        self.eps=eps
        self.eps1=eps1
        self.mu = mu
        # params correspond to the ws in the paper.
        self.model = model
        if self.model is None:
            self.params = self.zero()
        else:
            self.params = self.model.get_params() # this may be subject to change
        self.constraint = constraint
        #self.data=data
            
        self.w = self.f.reward.params #w is the weight of our model, see GIRL example
        
        h = lambda x: np.linalg.norm(x, 1) - 1  # sum of all the alphas must be 1
        self.alpha_cons = [{'type': 'eq', 'fun': h}]
        
        #Does not work, for some reason i is 99 for all lambda functions
# =============================================================================
#         for i in range(len(self.w)):
#             g = lambda x : x[i]
#             self.alpha_cons.append({'type':'ineq','fun':g})
#         
# =============================================================================
    
    def zero(self):
        self.f.zero()
        
    def reg(self,w):
        #for regularisation
        return 0
        
    def fit(self,trajs):
        start=True #for first iteration
        self.v = np.zeros(len(trajs)) #start
        v0 = np.random.rand(len(trajs))
        old_v = self.v
        
        Ms = self.f.compute_ms(trajs)
        
        ws = []
        
        loss = []
        while(not (self.v == np.ones(len(trajs))).all()): #find a termination condition perhaps double while (alternative search, and then decrement)
            print('New K value ///////////////////////////////////////////////////////////////////')
            self.v = np.zeros(len(trajs)) #start
            #Alternative search strategy
            while((start == True or not((old_v == self.v).all()) ) and np.sum(self.v)<len(trajs)):
                start=False
                #minimising for v
# =============================================================================
#                 result_v = opt.minimize(self.objective_v, v0, constraints=self.v_constraints)
#                 if not result_v.success:
#                     print(result_v.message)
#                     print(result_v)
#                 self.v = result_v.x
# =============================================================================
                print(np.linalg.norm(self.w,1))
                                
                losses = self.f.loss(self.f.reward.params,Ms)
                #second method use dirac
                old_v=self.v
                self.v = np.where(losses < 1/self.K,1.,0.)
                                
                print('ACS, ' + str(np.sum(self.v))+' samples taken in account')
                

                result_w = opt.minimize(self.objective_w, self.w,args=(Ms,),constraints=self.alpha_cons[0])
                if not result_w.success:
                    print(result_w.message)
                print(np.linalg.norm(result_w.x,1))
                self.w = result_w.x
                self.f.reward.set_params(self.w)
            
                ws.append(self.w)
            self.K=self.mu * self.K
        
        return ws
    
    def fit2(self,trajs):
        start=True #for first iteration
        self.v = np.zeros(len(trajs)) #start
        v0 = np.random.rand(len(trajs))
        old_v = self.v
        
        Js = self.f.compute_js(trajs)
        
        ws = []
        
        loss = []
        while(not (self.v == np.ones(len(trajs))).all()): #find a termination condition perhaps double while (alternative search, and then decrement)
            print('New K value ///////////////////////////////////////////////////////////////////')
            self.v = np.zeros(len(trajs)) #start
            start=True
            #Alternative search strategy
            old_w = self.w
            while((start == True or np.sum(self.v - old_v)>self.eps1 ) and np.sum(self.v)<len(trajs)):
                start=False
                #minimising for v
                result_v = opt.minimize(self.objective_v, self.v, args=(Js,), bounds = [(0,1)]*len(trajs))
                if not result_v.success:
                    print(result_v.message)
                    print(result_v)
                old_v = self.v
                self.v = result_v.x #check if we need process to go to [0,1]
                print(np.linalg.norm(self.w,1))
                                
                                
                print('ACS, ' + str(np.sum(self.v))+' samples taken in account')
                
                J = np.tensordot(self.v,Js,axes=([0],[0]))
                M = np.dot(J.T,J)

                result_w = opt.minimize(self.objective_w2, self.w,args=(M,),constraints=self.alpha_cons[0])
                if not result_w.success:
                    print(result_w.message)
                print(np.linalg.norm(result_w.x,1))
                self.w = result_w.x
                self.f.reward.set_params(self.w)
            if(np.linalg.norm(self.w - old_w)):
                ws.append(self.w)
            self.K=self.mu * self.K
        
        return ws
    
    def objective_w(self,w,Ms):
        return(np.dot(self.v, self.f.loss(w,Ms))+self.reg(w) ) #le reste est independant de w donc pas besoin de calculer
    
    def objective_w2(self,w,M):
        #compute the real loss as explained in Inverse Reinforcement Learning through Policy Gradient Minimization
        #main issue is self paced is defined as a sum of loss whereas GIRL is ||sum ... ||
        return self.f.loss2(w,M)+self.reg(w)
    
    def objective_v(self,v,Js):
        J = np.tensordot(v,Js,axes=([0],[0]))
        M = np.dot(J.T,J)
        return(self.f.loss2(self.w,M) - np.sum(v)/self.K) #think about a way to only calculate objective if v is 1
        
    def objective(self,inputs,trajs):
        w,v = inputs
        return(self.reg(w) + np.dot(v,self.f.loss(self.w,trajs)) - np.sum(v)/self.K)
        
    
        
   