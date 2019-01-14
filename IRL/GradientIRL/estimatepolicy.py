import numpy as np
from copy import copy
from tqdm import tqdm

class Quality():
    """
    A Quality value for state-action values to use in a Gibbs policy.
    """
    
    def __init__(self, theta=None):
        if theta is None:
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
        
    def set_theta(self, theta):
        self.theta = theta
    
    def add_theta(self, theta):
        self.theta += theta
    
    def grad(self, state, action):
        a = self._action_to_vector(action)
        v = state[1]
        return v * a
    
    def zero(self):
        return np.zeros(3)
        
    def value(self, state, action):
        a = self._action_to_vector(action)
        v = state[1]
        return np.dot(self.theta, v * a)

class GibbsPolicy():
    """
    The Gibbs policy is defined by :
    
    pi(a|s) = exp(K * Q(s,a))/sum_a'(exp(K * Q(s, a'))
    """
    
    def __init__(self, env, T, K, Q=None, gamma=0.9):
        self.K = K
        if Q is None:
            self.Q = Quality()
        else:
            self.Q = Q
        self.actionlist = [0, 1, 2]
        self.gamma = gamma
        self.env = env
        self.T = T
    
    def set_theta(self, theta):
        self.Q.set_theta(theta)
    
    def add_theta(self, theta):
        self.Q.add_theta(theta)
    
    def get_theta(self):
        return copy(self.Q.theta)
    
    def proba(self, state, action):
        """
        The probability of taking action when in state.
        """
        denom = 0
        for a in self.actionlist:
            denom += np.exp(self.K * self.Q.value(state, a))
        return np.exp(self.K * self.Q.value(state, action))/denom
    
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
    
    def done(self, state):
        pos = state[0]
        if pos >= 0.5:
            return True
        else:
            return False
    
    
    def grad_log(self, traj):
        """
        Computes the gradient of the log-probability according to theta, on a batch composed
        of a list of states and actions.
        """
        g = self.Q.zero()
        for state, action in traj:
            g+=self.Q.grad(state, action)
			for a in self.actionlist:
				g -= self.proba(state, a) * self.Q.grad(state, a)
		g = self.K * g
        return g
    
	def grad_log_state(self, state):
		#return all values for all actions
		g = self.Q.grad(state, action)
        for a in self.actionlist:
            g -= self.proba(state, a) * self.Q.grad(state, a)
		return g
	
    def grad_log_J(self, N, render=False):
        """
        Estimates, by a Monte-Carlo scheme on trajectories, the gradient of the objective 
        (score fonction) with respect to theta.
        """
        grad = self.Q.zero()
        length = 0
        for n in range(N):
            #print('episode')
            # performing N trajectories to compute the gradient estimate.
            done = False
            state = self.env.reset()  # reset the environment at the beginning of each MC run
            # initialize the gradient of log-probabilities and cumulative reward
            gradlog = self.Q.zero()
            R = 0.
            for t in range(self.T):
                if not self.done(state):
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
                    R += self.gamma**t * reward
                    # go into next state
                    state = next_state
                    length += 1/N
                else:
                    break
            grad += 1/N * gradlog * R
        return grad, length
    
    def learn(self, I, N, alphas):
        """
        Learn, by ascent on the gradient of the objective function with respect to theta,
        the optimal parameter for the policy.
        """
        thetas = []
        grads = []
        lengths = []
        for i in tqdm(range(I)):
            grad, l = self.grad_log_J(N)
            grads.append(grad)
            self.add_theta(alphas[i] * grad)
            theta = copy(self.get_theta())
            #print(theta)
            thetas.append(theta)
            lengths.append(l)
        return thetas, grads, lengths
    
    def episode(self, render=False):
        states = []
        actions = []
        rewards = []
        next_states = []
        state = self.env.reset()
        for t in range(self.T):
            if not self.done(state):
                if render:
                    self.env.render()
                action = self.sample(state)
                next_state, reward, _, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                state = next_state
            else:
                break
        print(t)
        return dict(states=states, actions=actions, rewards=rewards, next_states=next_states)
    
    def estim_theta(self, trajs, alphas, n_steps):
        """
        Performs gradient ascent on the log-likelihood to estimate the theta parameter.
        """
        trace = [self.get_theta()]
        N = len(trajs)
        for s in tqdm(range(n_steps)):
            grad = self.Q.zero()
            for traj in trajs:
                grad += 1/N * self.grad_log(traj)
            self.add_theta(alphas[s] * grad)
            trace.append(self.get_theta())
        return trace
    
    def fit(self, data, n_steps):
        """
        Interface to use to fit the theta parameter to given trajectories.
        
        n_steps = number of steps of gradient ascent.
        """
        N = len(data)
        trajs = []
        for i in range(N):
            traj = []           # building a single trajectory
            T = len(data[i]['states'])
            for t in range(T):
                state = data[i]['states'][t]
                action = data[i]['actions'][t]
                traj.append([state, action])
            trajs.append(traj)
        alphas = [.5 for i in range(n_steps)]
        thetas = self.estim_theta(trajs, alphas, n_steps)
        self.set_theta(thetas[-1])
        return thetas








































