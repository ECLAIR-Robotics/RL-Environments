import pybullet_envs.bullet.minitaur_gym_env as e

max_step = 1000
episode_count = 0
max_episodes = 50

env = e.MinitaurBulletEnv(render=True)

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
