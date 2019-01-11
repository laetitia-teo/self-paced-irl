# Helper class for parametric value functions

import numpy as np

class ParametricValueFunction():
    
    def __init__(self, n_psi):
        self.n_psi = n_psi
        self.phi = self.zero()
    
    def zero(self):
        return np.zeros(self.n_psi)
    
    def basis(self, state, i):
        raise NotImplementedError()
        value = 0
        return value # TODO
    
    def basis_grad(self, state, i):
        raise NotImplementedError()
        value = 0
        return value # TODO
    
    def value(self, state):
        value = 0
        for i, p in enumerate(self.phi):
            value += p * self.basis(state, i)
        return value
    
    def grad(self, state):
        value = 0
        for i, p in enumerate(self.phi):
            value += p * self.basis_grad(state, i)
        return value
