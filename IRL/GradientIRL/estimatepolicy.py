import numpy as np
from trajectory_generator.readtrajectory import read

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
    
    def __init__(self, K, Q=None):
        self.K = K
        if Q is None:
            self.Q = LinearQuality()
        else:
            self.Q = Q
        self.actions = [0, 1, 2]
    
    def set_theta(self, theta):
        self.Q.set_theta(theta)
    
    def proba(self, state, action):
        """
        The probability of taking action when in state.
        """
        denom = 0
        for a in self.actions:
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
        for action in self.actions:
            thresh += self.proba(state, action)
            if p <= thresh:
                return action
    
    def grad(self, batch):
        """
        Computes the gradient of the log-likelihood according to theta, on a batch composed
        of a list of states and actions.
        """
        g = np.zeros(6)
        for state, action in batch:
            g += self.K * self.Q.grad(state, action)
            for a in self.actions:
                g -= self.K * self.proba(state, a) * self.Q.grad(state, a)
        return g
    
    def estimate_theta_multibatch(self, batches, alphas):
        """
        Performs gradient ascent on the log-likelihood to estimate the theta parameter.
        """
        trace = [self.theta]
        for idx, batch in enumerate(batches):
            self.theta += alphas[idx] * self.grad(batch)
            trace.append[self.theta]
        return trace
    
    def estimate_theta_n(self, N, alphas, batch):
        """
        Performs gradient ascent on the log-likelihood to estimate the theta parameter.
        """
        trace = [self.theta]
        for i in range(N):
            self.theta += alphas[i] * self.grad(batch)
            trace.append[self.theta]
        return trace
    
    def fit(self, data, N):
        """
        Interface to use to fit the theta parameter to the data.
        """
        alphas = [1/(n+1) for n in range(N)]
        self.estimate_theta_n(N, alphas, data)








































