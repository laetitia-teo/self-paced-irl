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
        sp = env.observation_space
        self.dx = dx
        self.dv = dv
        self.lx = (sp.high[0] - sp.low[0])  # length of the position interval
        self.lv = (sp.high[1] - sp.low[1]) # length of the velocity interval
        self.zx = - sp.low[0]  # zero of the position interval
        self.zv = - sp.low[1] # zero of the velocity interval
        # tune sigma according to the discretization
        self.sigma_inv = inv(np.array([[0.5*(self.lx/self.dx)**2, 0.  ],
                                      [0., 0.5*(self.lv/self.dv)**2]])) 
        self.params = np.ones(dx * dv)
        #self.params = np.random.random_sample(dx*dv)
        self.params /=np.linalg.norm(self.params,1)
        
        self.centers = np.zeros((dx*dv,2))
        self.fill_centers()
# =============================================================================
#         self.params = np.zeros(dx*dv)
# =============================================================================
        self.env = env
        
    def fill_centers(self):
        for i in range(self.dx):
            self.centers[i*self.dv:(i+1)*self.dv,0] += i / (self.dx-1) * self.lx - self.zx 
        for j in range(self.dv):
            self.centers[j::self.dv,1] += j / (self.dv-1) * self.lv - self.zv
    
    def value(self, state, action):
# =============================================================================
#         r = 0.
#         for idx in range(self.dx*self.dv):
#             r += self.params[idx] * self.basis(state, idx)
# =============================================================================
        r=np.dot(self.params,self.basis2(state))
        return r
    
    def basis2(self,state):
        state_normalized = state - self.centers
        
        result = np.einsum('ij,ij->i', np.dot(state_normalized, self.sigma_inv), state_normalized)
        return np.exp(-result/2)
    
    def basis(self, state, idx):
        s = state
        si = self.centers[idx] 
        return np.exp(-np.dot((s - si), np.dot(self.sigma_inv, (s - si))))
    
    def partial_value(self, state, action, idx):
        j = idx % self.dv
        i = (idx - j)//self.dv
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
        fig = plt.figure()
        x = np.arange(-1.2, 0.6, 0.1)
        v = np.arange(-0.07, 0.07, 0.005)
        X = len(x)
        V = len(v)
        print(X)
        print(V)
        x, v = np.meshgrid(x, v)
        
        r = np.zeros([X, V])
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i in range(X):
            for j in range(V):
                xi = i / (X-1) * 1.8 - 1.2
                vj = j / (V-1) * 0.14 - 0.07
                r[i, j] = self.value([xi, vj], 1)
        ax.plot_surface(x, v, r.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

