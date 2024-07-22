import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import minari
from minari import DataCollector
from typing import Union

Activation = Union[str, nn.Module]

_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}

device = None

epochs = 5
learning_rate = 0.001

observation_size = 12
output_size = 1
hidden_size = 3
observations = []
actions = []
rewards = []


class Network(nn.Module):
    def __init__(self, input_size1, hidden_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# def build_mlp(
#               input_size1: int,
#               input_size2: int,
#               output_size: int,
#               n_layers: int,
#               hidden_size: int,
#               activation: Activation = "tanh",
#               output_activation: Activation = "identity",
#               ):
#     if isinstance(activation, str):
#         activation = _str_to_activation[activation]
#     if isinstance(output_activation, str):
#         output_activation = _str_to_activation[output_activation]
#     layers = []
#     for _ in range(n_layers):
#         layers.append(nn.Linear(input_size1 + input_size2, hidden_size))
#         layers.append(activation)
#     layers.append(nn.Linear(hidden_size, output_size))
#     layers.append(output_activation)
#
#     mlp = nn.Sequential(*layers)
#     mlp.to(device)
#     return mlp


def build_target_network():

    network = Network(observation_size, hidden_size, output_size)
    for p in network.parameters():
        p.requires_grad_(False)

    return network


def build_predictor_network():

    network = Network(observation_size, hidden_size, output_size)

    return network


def convert_dict_to_tensor(data_dict):

    for key, value in data_dict.items():
        if not isinstance(value, (int, float, list, np.ndarray)):
            raise ValueError(f"Unsupported value type for key '{key}': {type(value)}")

    # Flatten the dictionary values and convert to a list of arrays
    flattened_values = []
    for key, value in data_dict.items():
        if isinstance(value, (int, float)):
            flattened_values.append(np.array([value]))
        else:
            flattened_values.append(np.array(value).flatten())

    # Concatenate all flattened arrays
    concatenated_values = np.concatenate(flattened_values)

    # Convert the concatenated array to a PyTorch tensor
    tensor = torch.tensor(concatenated_values, dtype=torch.float32)

    return tensor


def convert_tuple_to_tensor(data_tuple, dtype=torch.float32):
    # Convert the tuple to a list
    data_list = list(data_tuple)

    # Convert the list to a PyTorch tensor
    tensor = torch.tensor(data_list, dtype=dtype)

    return tensor


def get_train_observation():

    env = gym.make('PointMaze_UMazeDense-v3')
    state = env.reset()
    train_observation = convert_tuple_to_tensor(state)
    return train_observation






