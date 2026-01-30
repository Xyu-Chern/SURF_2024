import os
os.add_dll_directory("C://Users//28449//.mujoco//mjpro150//bin")


import mujoco_py
import gymnasium as gym
# from sppo import PPO
from minari import DataCollector
from stable_baselines3 import A2C


env = DataCollector(gym.make("PointMaze_UMazeDense-v3", render_mode="rgb_array"))

print(env.reset())

model = A2C("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
model.replay_buffer = dataset

dataset = env.create_dataset(dataset_id="PointMaze_UMazeDense-v3_v2",
                             eval_env="PointMaze_UMazeDense-v3",
                             minari_version="3",
                             algorithm_name="ppo")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()





