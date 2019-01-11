# Wrapper class for the optimization problem

# This class will be used as the interface between the IRL part and the optim part
# Default model (initialized as None) results in a linear regression.

# Feel free to play around with different models before I come and try to plug in
# some IRL


import numpy as np
from numpy.linalg import norm
import scipy.optimize as opt

class OptimizationWrapper():
    
    def __init__(self, dim, K0, model=None):
        self.dim = dim
        self.K0 = K0
        self.model = model
        # params correspond to the ws in the paper.
        if self.model is None:
            self.params = self.zero()
        else
            self.params = self.model.get_params() # this may be subject to change
    
    def zero(self):
        return np.zeros(self.dim)
    
    def set_params(self, params):
        if self.model is None:
            self.params = params # for the default case
        else:
            self.model.set_params() 
    
    def regularization(self):
        return norm(self.params)**2
    
    def predict(self, x):
        # this is just a toy example in the default case
        if self.model:
            return self.model.predict(x)
        else:
            return np.dot(x, self.params) # default
    
    def loss(self, x, y):
        # for this example, we can take the MSE
        return (self.predict(x) - y)**2 + self.regularization()
    
    def self_paced_objective(self, xi, yi, K):
        raise NotImplementedError()
    
    def self_paced_learn(self, dataset, eps):
        raise NotImplementedError() # use opt.minimize(fun) or pytorch or whatever
    
