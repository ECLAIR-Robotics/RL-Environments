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
        Samples a random batch of indices from the replay buffer
        """
        return np.random.choice(len(self.memory), num_samples, replace=replace)

    def get_transitions(self, return_type, indices=None, device=None):
        """
        Gets all the transitions or just those at given indices

        Args:
            return_type (str): the type of the return value
            indices (Iterable[int]): the indices of the transitions to return
            device (torch.device): the device to return the transitions to
        
        Returns:
            state: len(indices) x state_dim iterable of states
            action: len(indices) x action_dim iterable of actions
            reward: len(indices) iterable of rewards
            next_state: len(indices) x state_dim iterable of next states
            done: len(indices) iterable of booleans indicating whether the episode has ended
            info: len(indices) x info_dim iterable of extra information
        """
        # Extract the transitions from indices
        if indices is None:
            transitions = list(self.memory)
        else:
            transitions = [self.memory[i] for i in indices]

        state, action, reward, next_state, done, info = zip(*transitions)
        
        # Data conversions
        if return_type == 'numpy':
            return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done), info
        elif return_type == "torch":
            return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32), torch.tensor(next_state, dtype=torch.float32), torch.tensor(done), info
        elif return_type == 'tuple':
            return state, action, reward, next_state, done, info
    
    def sample_transitions(self, num_samples, replace=False, return_type='numpy'):
        """
        Samples a batch of transitions from the replay buffer
        """
        sampled_indices = self.sample_indices(num_samples, replace)
        return self.get_transitions(return_type, sampled_indices=sampled_indices)

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
            info: extra information about the transition (e.g. log probs)
        """
        self.memory.append((state, action, reward, next_state, done, info))
        return
    
    def __len__(self):
        return len(self.memory)
    
if __name__ == "__main__":
    buffer = ReplayBuffer(10)
    for i in range(10):
        buffer.append(1, 2, 3, 4, True)
    state, action, reward, next_state, done, info = buffer.sample_transitions(3)
    print(state)
    print(action)
    print(reward)
    print(done)
    print(info)
    state, action, reward, next_state, done, info = buffer.get_transitions('torch')
    print(state)
    print(action)
    print(reward)
    print(done)
    print(info)