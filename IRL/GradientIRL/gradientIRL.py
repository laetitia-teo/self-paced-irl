import sys
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv, norm
from tqdm import tqdm
from copy import copy

sys.path.append('../..')

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
    
    def loss(self, trajs):
        M = np.dot(self.jacobian.T, self.jacobian)
        alpha = self.reward.params
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
    
            





































