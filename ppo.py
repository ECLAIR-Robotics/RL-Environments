from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import torch.optim as optim

class PPO:
    def __init__(self, hyperparameters):
        """
        Args:
            hyperparameters (dict): a dictionary of hyperparameters
        Returns:
            None
        """
        # Extract hyperparameters
        self.lr = hyperparameters['learning_rate']
        self.discount = hyperparameters['discount_rate']
        self.num_batch_transitions = hyperparameters['num_batch_transitions']
        self.state_dim = hyperparameters['state_dim']
        self.action_dim = hyperparameters['action_dim']
        self.total_train_steps = hyperparameters['total_train_steps']
        self.num_rand_epochs = hyperparameters['num_rand_epochs']
        # Initialize actor/critic networks
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        # Initialize replay buffer and environment
        self.replay_buffer = ReplayBuffer(self.batch_size)
        self.enviroment = KukaGymEnv(renders=True)

    def sample_action_space(self):
        """
        Uniformly samples an action from the action space
        """
        return self.enviroment.action_space.sample()

    def sample_policy(self, total_time_steps):
        """
        Uses the policy to interact with the environment and fill replay buffer
        """
        # Sample certain number of transitions
        num_samples = 0
        while num_samples < total_time_steps:
            done = False
            cur_state = self.enviroment.reset()
            # Run episodes until we get enough samples
            for i in range(10000000):
                # Sample action from the policy
                action, log_prob = self.actor.sample_action()
                # Take action in the environment
                next_state, reward, done, info = self.enviroment.step(action)
                # Store transition in replay buffer
                num_samples += 1
                self.replay_buffer.append(cur_state, action, reward, next_state, done, (log_prob))
                cur_state = next_state
                # End the environment if done 
                if done:
                    break

    def train(self):
        """
        Trains the actor and critic networks using PPO
        """
        print("Training PPO for {} steps".format(self.total_train_steps))
        num_time_steps = 0
        while num_time_steps < self.total_train_steps:
            # Sample policy to fill replay buffer
            self.sample_policy()
            state, action, reward, next_state, done, info = self.sample_transitions(return_type="torch_tensor")
            num_time_steps += 1
            # Compute and normalize advantage
            value = self.critic(state)
            next_value = self.critic(next_state)
            


    def test(self):
        return


        