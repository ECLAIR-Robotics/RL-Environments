import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, extra_params=None):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 