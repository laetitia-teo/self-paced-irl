# REINFORCE Agent for generating optimal trajectories

import numpy as np
from tqdm import tqdm

class LinearQuality():
    """
    A Linear Quality value for state-action values to use in a Gibbs policy.
    """
    
    def __init__(self, theta=None):
        try:
            assert(type(theta) == type(np.zeros(6)) and len(theta) == 6)
        except AssertionError:
            raise TypeError(
                'theta must be a numpy.ndarray of length 5, but is {}'.format(theta))
        if theta is None:
            self.theta = np.zeros(6)
        else:
            self.theta = theta
        
    def _action_to_vector(action):
        vector = np.zeros(3)
        if action in [0, 1, 2]:
            vector[action] = 1.
            return vector
        else raise TypeError(
            'action must be one of 0, 1, 2, but is {}'.format(action))
    
    def set_theta(self, theta):
        self.theta = theta
    
    def grad(self, state, action):
        return np.concatenate([state, a, [1]])
        
    def value(self, state, action):
        a = _action_to_vector(action)
        return np.dot(theta, np.concatenate([state, a, [1]])) # adding a bias

class GibbsPolicy():
    """
    The Gibbs policy is defined by :
    
    pi(a|s) = exp(K * Q(s,a))/sum_a'(exp(K * Q(s, a'))
    """
    
    def __init__(self, env, T, K, Q=None, gamma=0.9):
        self.K = K
        if Q is None:
            self.Q = LinearQuality()
        else:
            self.Q = Q
        self.actionlist = [0, 1, 2]
        self.gamma = gamma
        self.env = env
        self.T = T
    
    def set_theta(self, theta):
        self.Q.set_theta(theta)
    
    def proba(self, state, action):
        """
        The probability of taking action when in state.
        """
        denom = 0
        for a in self.actionlist:
            denom += np.exp(self.K * self.Q.value(state, a))
        return np.exp(self.K * self.Q.value(state, action)/denom
    
    def logproba(self, state, action):
        return np.log(self.proba(state, action))
    
    def sample(self, state):
        """
        Samples an action according to the probability conditioned on state.
        """
        thresh = 0
        p = np.random.random()
        for action in self.actionlist:
            thresh += self.proba(state, action)
            if p <= thresh:
                return action
    
    def gradlog(self, N, render=False):
        """
        Estimates, by a Monte-Carlo scheme on trajectories, the gradient of the objective 
        (score fonction) with respect to theta.
        """
        grad = np.zeros(6)
        for n in range(N):
            # performing N trajectories to compute the gradient estimate.
            done = False
            state = self.env.reset()  # reset the environment at the beginning of each MC run
            # initialize the gradient of log-probabilities and cumulative reward
            gradlog = np.zeros(6)
            R = 0.
            for t in range(self.T):
                if not done:
                    if render:
                        self.env.render()
                    # sample an action according to the policy
                    action = self.sample(state)
                    # step into the environment
                    next_state, reward, done, _ = self.env.step(action)
                    # updates the values of the gradient and of the cumulative rewards
                    gradlog += self.K * self.Q.grad(state, action)
                    for a in self.actionlist:
                        gradlog -= self.K * self.proba(state, a) * self.Q.grad(state, a)
                    R += self.gamma**idx * reward
                    # go into next state
                    state = next_state
            grad += 1/N * gradlog * R
        return grad
    
    def learn(self, I, N, alphas):
        """
        Learn, by ascent on the gradient of the objective function with respect to theta,
        the optimal parameter for the policy.
        """
        thetas = []
        for i in range(I):
            grad = gradlog(N)
            self.theta += alphas[i] * grad  
            thetas.append(self.theta)
        return thetas  
    





























