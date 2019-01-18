import numpy as np
from tqdm import tqdm
from copy import copy

class Quality:
    """
    Abstrct class for a Quality function, specifying all that has to be
    implemented.
    """
    def __init__(self, theta=None):
        self.theta = theta
    
    def set_theta(self, theta):
        self.theta = theta
    
    def add_theta(self, theta):
        self.theta += theta
        
    def zero(self):
        return np.zeros(len(self.theta))
        
    def grad(self, state, action):
        raise NotImplementedError()
    
    def value(self, state, action):
        raise NotImplementedError()

class CartPoleQuality(Quality):
    """
    Abstrct class for a Quality function, specifying all that has to be
    implemented.
    """
    def __init__(self, theta=None):
        self.theta = 2*(np.random.random(16) - 0.5*np.ones(16))
    
    def set_theta(self, theta):
        self.theta = theta
    
    def add_theta(self, theta):
        self.theta += theta
        
    def zero(self):
        return np.zeros(len(self.theta))
        
    def build_vector(self, state, action):
        f = np.zeros(12)
        ac = [0., 0., 0.]
        ac[action] = 1.
        for i in range(4):
            for j in range(3):
                f[i*3+j] = state[i]*ac[j]
        return np.concatenate([state, f])
    
    def grad(self, state, action):
        return self.build_vector(state, action)
    
    def value(self, state, action):
        features = self.build_vector(state, action)
        return np.dot(self.theta, features)

class BasicQuality(Quality):
    """
    A Simple Quality value for state-action values to use in a Gibbs policy.
    """
    
    def __init__(self, theta=None):
        if theta is None:
            # uniform on [-1, 1]
            self.theta = 2*(np.random.random(3) - 0.5*np.ones(3))
        else:
            self.theta = theta
        
    def _action_to_vector(self, action):
        vector = np.zeros(3)
        if action in [0, 1, 2]:
            vector[action] = 1.
            return vector
        else:
            raise TypeError(
                'action must be one of 0, 1, 2, but is {}'.format(action))
        
    def grad(self, state, action):
        a = self._action_to_vector(action)
        v = state[1]
        return v * a
    
    def value(self, state, action):
        a = self._action_to_vector(action)
        v = state[1]
        return np.dot(self.theta, v * a)
    
class FC1Quality(Quality):
    """
    A Quality value function based on a Fully-Connected neural network, with
    one hidden layer.
    """
    
    def __init__(self, n_h, input_size=3):
        # input is [position, velocity, action] 
        # number of hidden units
        self.n_h = n_h
        # weights between input and hidden units
        self.Wih = np.zeros([input_size, n_h])
        # weights between hidden units and output
        self.Who = np.zeros(n_h)
        
    def _action_to_vector(self, action):
        return
        
    def set_theta(self, theta):
        self.theta = theta
    
    def add_theta(self, theta):
        raise NotImplementedError()
    
    def grad(self, state, action):
        raise NotImplementedError()
    
    def zero(self):
        raise NotImplementedError()
        
    def value(self, state, action):
        raise NotImplementedError()

class RBFQuality(Quality):
    """
    A Quality value function based on a linear combination of Gaussian RBFs in
    the state-action space.
    """
    
    def __init__(self, dx, dv, theta=None):
        if theta is None:
            self.theta = np.zeros(dx*dv)
        else:
            self.theta = theta
        self.dx = dx
        self.dv = dv
        self.lx = 1.8  # length of the position interval
        self.lv = 0.14 # length of the velocity interval
        self.zx = 0.6  # zero of the position interval
        self.zv = 0.07 # zero of the velocity interval
        # tune sigma according to the discretization
        self.sigma_inv = inv(np.array([[.25/dx, 0.  ],
                                      [0., .00015/dv]])) 
                              
    def basis(self, state, action, idx):
        #TODO : add influence of the action
        j = idx % self.dv
        i = (idx - j)/self.dv
        x, v = state
        xi = i / (self.dx-1) * self.lx - self.zx 
        vj = j / (self.dv-1) * self.lv - self.zv
        s = np.array([x, v])
        si = np.array([xi, vj])
        return np.exp(-np.dot((s - si), np.dot(self.sigma_inv, (s - si))))
    
    def partial_value(self, state, action, idx):
        return self.theta[idx] * self.basis(state, action, idx)
    
    def value(self, state, action):
        value = 0
        for i in range(len(self.theta)):
            value += self.partial_value(state, action, i)
        return value
    
    def grad(self, state, action):
        """
        Computes the gradient with respect to the value parameters, evaluated at state.
        """
        g = self.zero()
        for i, _ in enumerate(self.theta):
            g[i] = self.basis(state, action, i)
        return g


































