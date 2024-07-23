import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import minari
from minari import DataCollector
from typing import Union
import Help
import Network

env = gym.make('PointMaze_UMazeDense-v3')
# sample state s_0~p_0(s_0)
state = env.reset()
# obs = Network.get_train_observation()
# obs_shape = obs.reshape(1, 8)
# print(obs.dim())
# print(obs_shape.dim())
action = env.action_space.sample()
next_state, reward, done, truncated, info = env.step(action)
# ex_dict = Help.convert_state_to_tensor(next_state)
print(next_state)