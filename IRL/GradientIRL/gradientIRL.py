import sys
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv, norm
from tqdm import tqdm
from copy import copy

sys.path.append('../..')

class Reward():
    """
    Reward is the class defining a reward function for the IRL problem.
    Reward is a linear combination of (Gaussian) radial basis functions.
    
    dx -> number of basis functions on the position dimension;
    dv -> number of basis functions on the velocity dimension.
    """
    def __init__(self, dx, dv):
        self.dx = dx
        self.dv = dv
        self.lx = 1.8  # length of the position interval
        self.lv = 0.14 # length of the velocity interval
        self.zx = 0.6  # zero of the position interval
        self.zv = 0.07 # zero of the velocity interval
        # tune sigma according to the discretization
        self.sigma_inv = inv(np.array([[.05, 0.  ],
                                      [0., .0003]])) 
        self.params = np.zeros(dx * dv)
    
    def value(self, state, action):
        r = 0.
        for i in range(self.dx):
            for j in range(self.dv):
                r += self.params[i, j] * self.basis(state, i, j)
    
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

class GIRL():
    """
    A class for estimating the parameters of the reward given some trajectory data.
    """
    def __init__(self, reward, data, expert_policy):
        self.reward = reward   
        self.N = len(data)
        self.trajs = []
        for i in range(self.N):
            traj = []           # building a single trajectory
            T = len(data[i]['states'])
            for t in range(T):
                state = data[i]['states'][t]
                action = data[i]['actions'][t]
                traj.append([state, action])
            self.trajs.append(traj)
        self.expert_policy = expert_policy
        self.jacobian = np.zeros([len(expert_policy.get_theta()), len(reward.params)])
        
    def zero(self):
        """
        The zero vector in the space of policy parameters.
        """
        return self.expert_policy.Q.zero()
    
    def compute_gradient(self, idx):
        """
        Computes, averaged on the set of trajectories given to the GIRL object, the gradient of 
        the objective function with respect to the expert policy parameters associated with
        the l-th component of the reward.
        """
        grad = self.zero()
        for traj in self.trajs:
            g = self.expert_policy.grad_log(traj)
            r = self.reward.basis_traj(traj, idx)
            grad += r * g
        return grad/self.N
        
    def compute_jacobian(self):
        """
        Computes the Jacobian of the full objective function.
        """
        for l in tqdm(range(len(self.reward.params))):
            self.jacobian[:, l] = self.compute_gradient(l)
    
    def print_jacobian(self):
        with open('data.txt', 'a') as f:
            f.write(str(self.jacobian))
    
    def objective(self, alpha):
        M = np.dot(self.jacobian.T, self.jacobian)
        return np.dot(alpha, np.dot(M, alpha))
    
    def solve(self):
        # Define constraints
        h = lambda x: norm(x, 1) - 1  # sum of all the alphas must be 1
        eq_cons = {'type': 'eq', 'fun': h}
        #ineq_cons = {'type': '
        # Define starting point
        alpha0 = copy(self.reward.params)  # TODO : test with random starting vector
        # Define hessian and gradient of the objective?
        result = opt.minimize(self.objective, alpha0, constraints=eq_cons)
        if not result.success:
            print(result.message)
            print(result)
        alpha = result.x
        return alpha
    
            





































