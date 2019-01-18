import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import gym
e = gym.make('CartPole-v0')



class Reward():
    """
    Reward is the class defining a reward function for the IRL problem.
    Reward is a linear combination of (Gaussian) radial basis functions.
    
    dx -> number of basis functions on the position dimension;
    dv -> number of basis functions on the velocity dimension.
    """
    def __init__(self, d, env):
        sp = env.observation_space
        n = sp.shape[0]
        self.d = d*np.ones(n)
        self.l = (sp.high - sp.low)  # length of the position interval
        self.z = - sp.low  # zero of the position interval
        self.high = sp.high
        self.low = sp.low
        # tune sigma according to the discretization
        self.sigma_inv = inv(np.diag(1/self.d)**2)
        self.params = np.ones(d**n)
        self.params /= np.linalg.norm(self.params,1)
        
        s = 'np.mgrid['
        for i in range(n):
            s += ':d, '
        s = s[:-2]
        s += ']'
        m = eval(s)
        self.means = np.reshape(1/d*np.stack(m, axis=-1), (-1, n))
        self.env = env
        
    def to_01(self, state):
        return (state-self.low)/self.l
    
    def from_O1(self, state):
        return self.low + state*self.l
    
    def value(self, state, action):
        r=np.dot(self.params,self.basis(state))
        return r
    
    def basis(self,state):
        state_normalized = self.to_01(state) - self.means
        
        result = np.einsum('ij,ij->i', np.dot(state_normalized, self.sigma_inv), state_normalized)
        return np.exp(-result/2)
    
    def partial_value(self, state, action, idx):
        return self.params[idx] * self.basis(state, idx)
    
    def partial_traj(self, traj, idx):
        r = 0.
        for state, action in traj:
            r += self.partial_value(state, action)
        return r
    
    def basis_traj(self, traj):
        r = np.zeros(len(self.params))
        for state, _ in traj:
            r += self.basis(state) # no discount !
        return r
    
    def set_params(self, params):
        self.params = params
        
    def import_from_file(self, file_path):
        with open(file_path) as f:
            self.set_params(eval('np.'+f.read()))
    
    def export_to_file(self, file_path):
        with open(file_path, 'w') as f:
            f.write(repr(self.params))
    
    def plot(self):
        X = 50
        V = 50
        
        sp = self.env.observation_space
        
        x = np.linspace(sp.low[0], sp.high[0], X)
        v = np.linspace(sp.low[1], sp.high[1],V)

        x, v = np.meshgrid(x, v)
        
        r = np.zeros([X, V])
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i in range(X):
            for j in range(V):
                xi = i / (X-1) * (sp.high[0] - sp.low[0]) + sp.low[0]
                vj = j / (V-1) * (sp.high[1] - sp.low[1]) + sp.low[1]
                r[i, j] = self.value([xi, vj], 1)
        # =============================================================================
        #         r[i,j] = reward.basis([xi,vj],0)
        # =============================================================================
        ax.plot_surface(x, v, r.T, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

