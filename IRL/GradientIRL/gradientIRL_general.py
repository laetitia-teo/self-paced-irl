import sys

sys.path.append('../..')
sys.path.append('..')

import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv, norm
from tqdm import tqdm
from copy import copy
from IRL import IRL


class GIRL(IRL):
    """
    A class for estimating the parameters of the reward given some trajectory data.
    """
    def __init__(self, reward, expert_policy):
        self.reward = reward   #untrained reward
        self.expert_policy = expert_policy
        self.jacobian = np.zeros([len(expert_policy.get_theta()), len(reward.params)])
        
    def zero(self):
        """
        The zero vector in the space of policy parameters.
        """
        return self.expert_policy.Q.zero()
    
    def compute_gradient(self, idx, trajs):
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
        
    def compute_jacobian(self, trajs):
        """
        Computes the Jacobian of the full objective function.
        """
        jacobian = np.zeros([len(self.expert_policy.get_theta()), len(self.reward.params)])

        for traj in tqdm(trajs):
            g = self.expert_policy.grad_log(traj)
            R = self.reward.basis_traj(traj)
            jacobian += np.outer(g, R)
        jacobian /= len(trajs)
        return jacobian
    
    def print_jacobian(self):
        with open('data.txt', 'a') as f:
            f.write(str(self.jacobian))
    
    def loss2(self, alpha, M):
        return np.dot(alpha, np.dot(M, alpha))
    
    def loss(self, w, Ms):
        losses = []
        for i in range(len(Ms)):
            losses.append(self.loss2(w,Ms[i]))
        return np.asarray(losses)

    def loss3(self, w, Ms):
        return(np.sum(self.loss(w,Ms),axis=0))
    
    def compute_ms(self, trajs):
        jacobians = self.compute_js(trajs)
        Ms=np.asarray(jacobians.shape[0],jacobians.shape[2],jacobians.shape[2])
        for i in range(len(trajs)):
            Ms[i] = np.dot(jacobians[i].T, jacobians[i])
        return Ms
    
    def compute_js(self, trajs):
        jacobians = []
        for traj in tqdm(trajs):
            g = self.expert_policy.grad_log(traj)
            temp = np.zeros([len(self.expert_policy.get_theta()), len(self.reward.params)])
            for idx in range(len(self.reward.params)):
                temp[:,idx] = self.reward.basis_traj(traj, idx) * np.ones(len(temp))
            jacobian = (g*temp.T).T
            jacobians.append(jacobian)
        return np.asarray(jacobians)/len(trajs)
            
    def solve(self, trajs):
        # Define constraints
        h = lambda x: np.sum(x) + 1  # sum of all the alphas must be 1
        eq_cons = {'type': 'eq', 'fun': h}
        # Define starting point
        alpha0 = copy(self.reward.params)  
        
        jacobian = self.compute_jacobian(trajs)
        M = np.dot(jacobian.T, jacobian)
        
        result = opt.minimize(self.loss2, alpha0, args=(M,), constraints=eq_cons)
        
# =============================================================================
#         Ms = self.compute_ms(trajs)
#         result = opt.minimize(self.loss3, alpha0, args=(Ms,), constraints=eq_cons)
# =============================================================================
        
        if not result.success:
            print(result.message)
            print(result)
        alpha = result.x
        return alpha
    
            





































