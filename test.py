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


env = gym.make('PointMaze_UMazeDense-v3')
state = env.reset()
grt = [state]

print(grt)