from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
import gym

max_step = 1000
episode_count = 0
max_episodes = 50

env = CartPoleBulletEnv(renders=True)
observation = env.reset()
while episode_count < max_episodes:
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  # Rest environment if done
  if done:
    observation = env.reset()
    episode_count += 1
env.close()
