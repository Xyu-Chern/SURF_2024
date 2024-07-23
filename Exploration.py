import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import minari
from minari import DataCollector
from typing import Union
import Network
import Help

Num_rollouts = 10
Num_opt = 10
Len_rollout = 10
state_norm_steps = 4

t = 0


class Normalizer:
    def __init__(self, num_features, epsilon=1e-8):
        self.mean = np.zeros(num_features)
        self.var = np.ones(num_features)
        self.epsilon = epsilon
        self.count = 0

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)

    def update(self, x):
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.var = np.zeros_like(x)
        else:
            old_mean = self.mean.copy()
            self.mean += (x - self.mean) / self.count
            self.var += (x - old_mean) * (x - self.mean)


def calculate_intrinsic_reward(input_state: torch.Tensor,
                               target_network: nn.Module,
                               predictor_network: nn.Module):

    f_star = target_network.forward(input_state)
    f_theta = predictor_network.forward(input_state)
    intrinsic_reward = (abs(f_theta - f_star))**2
    return intrinsic_reward


# set normalizer
state_normalizer = Normalizer(4)
reward_normalizer = Normalizer(1)



# Train networks
target_network = Network.build_target_network()
predictor_network = Network.build_predictor_network()
optimizer = optim.Adam(predictor_network.parameters(), lr=0.001)

# Train loop:
for epochs in range(100):

    optimizer.zero_grad()
    train_observation = Network.get_train_observation()
    f_star = target_network(train_observation)
    logits = predictor_network(train_observation)

    loss = F.cross_entropy(logits, f_star)
    loss.backward()
    optimizer.step()


env = gym.make('PointMaze_UMazeDense-v3')


# sample state s_0~p_0(s_0)
state = env.reset()
initial_state = env.observation_space.sample()
state = initial_state
# update normalization parameters for observations
for _ in range(1, state_norm_steps + 1):
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    state_normalizer = Normalizer(4)
    state_normalizer.update(next_state)
    state = next_state


batches = {i + 1: [] for i in range(Num_rollouts)}
state = env.reset()
for i in range(1, Num_rollouts + 1):
    for j in range(1, Len_rollout + 1):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        intrinsic_reward = calculate_intrinsic_reward(Help.convert_dict_to_tensor(next_state),
                                                      target_network,
                                                      predictor_network)
        batches[i].append((state, next_state, action, reward, intrinsic_reward))























