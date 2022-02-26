from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer

class PPO:
    def __init__(self, hyperparameters):
        self.lr = hyperparameters['learning_rate']
        self.discount = hyperparameters['discount_rate']
        self.batch_size = hyperparameters['batch_size']
        self.state_dim = hyperparameters['state_dim']
        self.action_dim = hyperparameters['action_dim']

        self.actor = actor.Actor(self.state_dim, self.action_dim, self.lr)
        self.critic = critic.Critic(self.state_dim, self.action_dim, self.lr)
        self.replay_buffer = ReplayBuffer(self.batch_size)
    
    def train(self):
        
