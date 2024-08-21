import minari
import gymnasium as gym
from minari import DataCollector

env = gym.make('PointMaze_UMazeDense-v3')
env = DataCollector(env, max_buffer_steps=10000)

total_timesteps = 10240
env.reset(seed=123)
for _ in range(10):
    for _ in range(1024):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset(seed=123)

dataset = env.create_dataset(dataset_id="PointMaze_UMazeDense-v3_rand_v0",
                             eval_env="PointMaze_UMazeDense-v3",
                             minari_version="3",
                             algorithm_name="random")