# Helper class for parametric value functions

import numpy as np
from numpy.linalg import inv
from copy import copy

class RBFValueFunction():
    
    def __init__(self, dx, dv):
        self.dx = dx   # discretization on position axis
        self.dv = dv   # discretization on velocity axis
        self.lx = 1.8  # length of the position interval
        self.lv = 0.14 # length of the velocity interval
        self.zx = 0.6  # zero of the position interval
        self.zv = 0.07 # zero of the velocity interval
        self.n_psi = dx * dv
        self.sigma_inv = inv(np.array([[.05, 0.  ],
                                      [0., .0003]]))  # standard deviations
        self.phi = self.zero()
    
    def set_phi(self, phi):
        self.phi = phi
    
    def get_phi(self):
        return copy(self.phi)
    
    def zero(self):
        return np.zeros(self.n_psi)
    
    def basis(self, state, idx):
        j = idx % self.dv
        i = (idx - j)/self.dv
        x, v = state
        xi = i / (self.dx-1) * self.lx - self.zx 
        vj = j / (self.dv-1) * self.lv - self.zv
        s = np.array([x, v])
        si = np.array([xi, vj])
        return np.exp(-np.dot((s - si), np.dot(self.sigma_inv, (s - si))))
    
    def partial_value(self, state, idx):
        return self.phi[idx] * self.basis(state, idx)
    
    def value(self, state):
        value = 0
        for i in range(len(self.phi)):
            value += self.partial_value(state, i)
        return value
    
    def grad(self, state):
        """
        Computes the gradient with respect to the value parameters, evaluated at state.
        """
        g = self.zero()
        for i, _ in enumerate(self.phi):
            g[i] = self.basis(state, i)
        return g










































