# Discrete Q Agent for generating optimal trajectories


import gym
import numpy as np
from tqdm import tqdm


class QTable(dict):
    def __init__(self, default):
        self.default = default
        dict.__init__(self)    
    
    def __getitem__(self, idx):
        self.setdefault(idx, self.default)
        return dict.__getitem__(self, idx)

class QAgent():
    
    def __init__(self, env, T, discr=100, render=True, alpha=0.1, gamma=0.9):
        self.env = env
        self.qtable = QTable(0) #
        self.discr = discr
        self.state = env.reset()
        self.T = T
        self.gamma = gamma
        self.actionlist = [0, 1, 2]
        self.alpha = alpha
    
    def discretize_state(self, state):
        d_pos = np.floor(state[0]*self.discr)
        d_spd = np.floor(state[1]*self.discr)
        return d_pos, d_spd
    
    def get_Q(self, state, action):
        # discretization of position and velocity
        d_pos, d_spd = self.discretize_state(state)
        # checking for the best action according to our current q-function
        q = self.qtable[(d_pos, d_spd, action)]
        return q
    
    def set_Q(self, state, action, value):
        # discretization of position and velocity
        d_pos, d_spd = self.discretize_state(state)
        # checking for the best action according to our current q-function
        self.qtable[(d_pos, d_spd, action)] = value
    
    def get_max_Q(self, state):
        a0 = np.random.choice(self.actionlist)
        maxq = self.get_Q(state, a0)
        for a in self.actionlist:
            q = self.get_Q(state, a)
            if (q > maxq):
                maxq = q
        return maxq
    
    def get_max_action(self, state):
        besta = np.random.choice(self.actionlist)
        maxq = self.get_Q(state, besta)
        for a in self.actionlist:
            q = self.get_Q(state, a)
            if (q > maxq):
                maxq = q
                besta = a
        return besta
    
    def eps_greedy_action(self, eps, state):
        # epsilon greedy strategy : random action with probability epsilon
        # epsilon should be close to one in the first iterations, so as to reach the terminal
        # state
        if (np.random.random() < eps):
            return np.random.choice(self.actionlist)
        else:
            return self.get_max_action(state)
    
    def Q_update(self, state, action, nextstate, rew):
        q = self.get_Q(state, action)
        maxq = self.get_max_Q(nextstate)
        nextq = q + self.alpha*(rew + self.gamma*maxq - q)
        self.set_Q(state, action, nextq)
    
    def done(self, state):
        done = (np.floor(state[0]*self.discr) >= np.floor(0.5*self.discr))
        return done
        
    def episode(self, eps, render=False):
        # performing one trajectory, and performing online updates to the q-function
        done = False
        state = self.env.reset()
        state_trace = [state]
        states = []
        actions = []
        rewards = []
        next_states = []
        for t in range(self.T):
            if not self.done(state): 
                if render:
                    self.env.render()
                # choose an action
                action = self.eps_greedy_action(eps, state)
                # take a step, collect a reward
                next_state, reward, _, _ = self.env.step(action)
                # update q function
                self.Q_update(state, action, next_state, reward)
                # save transition into trajectory
                states.append(list(state))
                actions.append(action)
                rewards.append(reward)
                next_states.append(list(next_state))
                # go into next state
                state = next_state
            else:
                break
        return dict(states=states, actions=actions, rewards=rewards, next_states=next_states)
    
    def q_learn(self, N):
        # performing N trajectories
        #epsilon = [0.2 for i in range(int(N/2))] + [0.2/(i+1) for i in range(int(N/2))]
        #epsilon = [1/(i+1) for i in range(int(N))]
        epsilon = [0.1 for i in range(N)]
        for n in tqdm(range(N)):
            #print('episode {}'.format(n))
            self.episode(epsilon[n])
        # final episode
        self.episode(0.0, render=True)
    
    def generate_trajectories(self, n_traj):
        # with or without q updates ?
        traj = []
        for i in range(n_traj):
            traj.append(self.episode(0.0))
        return traj
    
    def generate_trajectory_file(self, n_traj):
        traj = self.generate_trajectories(n_traj)
        with open('data.txt', 'w') as f:
            for t in traj:
                f.write(str(t) + '\n')











































