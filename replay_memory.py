import random
import numpy as np

class ReplayMemory:
    
    def __init__(self, max_capacity):
        self.initial_states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.curr_capacity = 0
        self.max_capacity = max_capacity

    
    def insert(self, initial_state, action, reward, next_state, done):
        if self.curr_capacity != self.max_capacity:
            self.initial_states.append(initial_state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.curr_capacity += 1
        else:
            self.initial_states.append(initial_state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            
            self.initial_states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            



    
    def sample(self, n_samples):
        #need to return a list of samples that is n_samples long
        indexes = np.random.choice(self.curr_capacity, n_samples)
        initial_state_samples = np.array([self.initial_states[i] for i in indexes])
        action_samples = np.array([self.actions[i] for i in indexes])
        reward_samples = np.array([self.rewards[i] for i in indexes])
        next_state_samples = np.array([self.next_states[i] for i in indexes])
        done_samples = np.array([self.dones[i] for i in indexes])

        return initial_state_samples, action_samples, reward_samples, next_state_samples, done_samples