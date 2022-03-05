from collections import deque
from distutils.log import INFO
import numpy as np
import torch

class ReplayBuffer():
    """
    Implements an experience replay buffer used in many off-policy RL algorithms.
    Currently, the internal data structure is a deque, which is a double-ended queue.
    
    The dequeue will automatically pop the oldest transitions when the size of the dequeue is exceeded.
    """
    def __init__(self, max_transitions):
        """
        Contructor

        Args:
            max_transitions: the maximum number of transitions the buffer can hold
        """
        self.memory = deque(maxlen=max_transitions)
        return

    def sample_indices(self, num_samples, replace):
        """
        Samples a batch of indices from the replay buffer
        """
        return np.random.choice(len(self.memory), num_samples, replace=replace) 

    def get_transitions(self, sampled_indices, return_type):
        """
        Gets the transitions at the given indices
        """
        # Extract the transitionse
        transitions = [self.memory[i] for i in sampled_indices]
        state, action, reward, next_state, done, info = zip(*transitions)
        
        # Data conversions
        if return_type == 'numpy':
            return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done), info
        elif return_type == "torch_tensor":
            return torch.tensor(state), torch.tensor(action), torch.tensor(reward), torch.tensor(next_state), torch.tensor(done), info
        elif return_type == 'tuple':
            return state, action, reward, next_state, done, info
    
    def sample_transitions(self, num_samples, replace=False, return_type='numpy'):
        """
        Samples a batch of transitions from the replay buffer
        """
        sampled_indices = self.sample_indices(num_samples, replace)
        return self.get_transitions(sampled_indices, return_type)
    
    def clear(self):
        """
        Clears all the data from the buffer
        """
        self.memory.clear()
        return
    
    def append(self, state, action, reward, next_state, done, info=None):
        """
        Adds a new transition to the replay buffer

        Args:
            state: the current state of the environment
            action: the action taken in the current state
            reward: the reward received after taking the action
            next_state: the next state of the environment after taking the action
            done: whether the episode has ended after taking the action
        """
        self.memory.append((state, action, reward, next_state, done, info))
        return
    
    def __len__(self):
        return len(self.memory)
    
if __name__ == "__main__":
    buffer = ReplayBuffer(10)
    for i in range(10):
        buffer.append(1, 2, 3, 4, 5)
    state, action, reward, next_state, done = buffer.sample(3)
    print(state)
    print(action)