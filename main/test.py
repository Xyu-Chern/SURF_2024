import os
os.add_dll_directory("C://Users//28449//.mujoco//mjpro150//bin")
import sys
sys.path.append('.')

import gymnasium as gym
from utils.replaybuffer import ReplayBuffer

env = gym.make("PointMaze_UMazeDense-v3", render_mode="human")


memory = ReplayBuffer("PointMaze_UMazeDense-v3_v1")

from stable_baselines3 import A2C

print(env.reset())


model = A2C("MultiInputPolicy", env, verbose=1)
model.replay_buffer = memory

print("okkkk")
print(model.replay_buffer )