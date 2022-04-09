from architectures.actor import Actor
from architectures.critic import Critic
from utils.replay_buffer import ReplayBuffer
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import torch
import torch.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal  # TODO: Change this to a custom more stable normal distribution

class PPO:
    def __init__(self, hyperparameters):
        """
        Args:
            hyperparameters (dict): a dictionary of hyperparameters
        discounted_returns:
            None
        """
        # Extract hyperparameters
        self.lr = hyperparameters['learning_rate']
        self.discount = hyperparameters['discount_rate']
        self.num_batch_transitions = hyperparameters['num_batch_transitions']
        self.state_dim = hyperparameters['state_dim']
        self.action_dim = hyperparameters['action_dim']
        self.total_train_steps = hyperparameters['total_train_steps']
        self.max_episode_length = hyperparameters['max_episode_length']
        self.num_train_epochs = hyperparameters['num_train_epochs']
        self.device = hyperparameters['device']
        # Initialize actor/critic networks
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        # Initialize replay buffer and environment
        self.replay_buffer = ReplayBuffer(self.batch_size)
        self.enviroment = KukaGymEnv(renders=True)

    def sample_policy(self, total_time_steps=None):
        """
        Uses the policy to interact with the environment and fills the replay buffer
        
        Args:
            total_time_steps (int): the minimum total number of time steps to sample
        """
        if total_time_steps is None:
            total_time_steps = self.num_batch_transitions
        # Sample until at least have total_time_steps samples
        num_samples = 0
        while num_samples < total_time_steps:
            done = False
            cur_state = self.enviroment.reset()
            # Run full episodes until we get enough samples
            for i in range(self.max_episode_length):
                # Sample and perform action from the policy
                action, log_prob = self.actor.sample_action()
                next_state, reward, done, info = self.enviroment.step(action)
                # Store transition in replay buffer
                num_samples += 1
                self.replay_buffer.append(cur_state, action, reward, next_state, done, log_prob)
                cur_state = next_state
                if done:
                    break
        
    def rewards_to_go(self, sampled_rewards, sampled_dones, normalize=False):
        """
        Computes discounted returns for each episode from the rewards with the option of normalizing them 

        Args:
            sampled_rewards (Iterable[Float]): an iterable containing the rewards
            sampled_dones (Iterable[Bool]): an iterable containing whether there was terminal time step
            normalize (Bool): whether to normalize the returns
        """

        discounted_returns = []
        disc_return = 0
        # Compute discounted return for each episode
        for reward, done in zip(reversed(sampled_rewards), reversed(sampled_dones)):
            if done:
                disc_return = 0
            disc_return = reward + self.discount * disc_return
            discounted_returns.insert(0, disc_return) # TODO: Try to append and then reverse
        
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32).to(self.device)
        # Normalize discounted_returns
        if normalize:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)
        return discounted_returns
    
    def train(self):
        """
        Trains the actor and critic networks using PPO
        """
        print("Training PPO for {} steps".format(self.total_train_steps))
        num_time_steps = 0
        num_train_iterations = 0

        while num_time_steps < self.total_train_steps:
            # Sample policy for transitions
            self.sample_policy()
            state, action, reward, next_state, done, action_log_prob = self.sample_transitions(return_type="torch_tensor")
            # Compute advantage and value
            returns = self.rewards_to_go(reward, done)
            value = self.critic(state, action)
            advantage = returns - value.detach()
            # Normalize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            # Update actor and critic networks
            for train_epoch in range(self.num_train_epochs):
                # Compute value and action prob ratio
                value = self.critic(state, action) # TODO: See if you can just resuse value outside for loop
                action, cur_action_log_prob = self.actor.sample_action() 
                action_prob_ratio = torch.exp(cur_action_log_prob - action_log_prob)
                # Compute actor loss
                loss1 = action_prob_ratio * advantage
                loss2 = torch.clamp(action_prob_ratio, 1-self.clip, 1+self.clip ) * advantage
                actor_loss = -torch.min(loss1, loss2).mean()
                # Compute critic loss
                critic_loss = F.mse_loss(returns, value) 
                # Update actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True) # retain_graph prevents gradients from being erased after backprop
                self.actor_optim.step()
                # Update critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            
            self.buffer.clear()
            num_time_steps += len(state)

    def test(self):
        return


        