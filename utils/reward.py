import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Reward():
    """
    Reward is the class defining a reward function for the IRL problem.
    Reward is a linear combination of (Gaussian) radial basis functions.
    
    dx -> number of basis functions on the position dimension;
    dv -> number of basis functions on the velocity dimension.
    """
    def __init__(self, dx, dv, env):
        self.dx = dx
        self.dv = dv
        self.lx = env.low[0]  # length of the position interval
        self.lv = env.low[1] # length of the velocity interval
        self.zx = env.high[0]  # zero of the position interval
        self.zv = env.high[1] # zero of the velocity interval
        # tune sigma according to the discretization
        self.sigma_inv = inv(np.array([[.05, 0.  ],
                                      [0., .0003]])) 
        self.params = np.zeros(dx * dv)
    
    def value(self, state, action):
        r = 0.
        for idx in range(self.dx*self.dv):
            r += self.params[idx] * self.basis(state, idx)
        return r
    
    def basis(self, state, idx):
        j = idx % self.dv
        i = (idx - j)/self.dv
        x, v = state
        xi = i / (self.dx-1) * self.lx - self.zx 
        vj = j / (self.dv-1) * self.lv - self.zv
        s = np.array([x, v])
        si = np.array([xi, vj])
        return np.exp(-np.dot((s - si), np.dot(self.sigma_inv, (s - si))))
    
    def partial_value(self, state, action, idx):
        j = idx % self.dv
        i = (idx - j)/self.dv
        return self.params[idx] * self.basis(state, i, j)
    
    def partial_traj(self, traj, idx):
        r = 0.
        for state, action in traj:
            r += self.partial_value(state, action, idx)
        return r
    
    def basis_traj(self, traj, idx):
        r = 0.
        for state, _ in traj:
            r += self.basis(state, idx)
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
        ax = fig.gca(projection='3d')
        for i in range(self.dx):
        for j in range(self.dv):
            xi = i / (X-1) * 1.8 - 0.6
            vj = j / (V-1) * 0.14 - 0.07
            r[i, j] = reward.value([xi, vj], 1)
            ax.plot_surface(x, v, r.T, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

