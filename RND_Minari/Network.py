import torch
import torch.nn as nn

import numpy as np

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
learning_rate = 0.0001


output_size = 32
hidden_size1 = 32
hidden_size2 = 32
observations = []
actions = []
rewards = []


class Network(nn.Module):
    def __init__(self, input_size1, hidden_size_1, hidden_size_2, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size1, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_2)
        self.fc4 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def build_mlp(
              input_size1: int,
              input_size2: int,
              output_size: int,
              n_layers: int,
              hidden_size: int,
              activation: Activation = "tanh",
              output_activation: Activation = "identity",
              ):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(input_size1 + input_size2, hidden_size))
        layers.append(activation)
    layers.append(nn.Linear(hidden_size, output_size))
    layers.append(output_activation)

    mlp = nn.Sequential(*layers)
    mlp.to(device)
    return mlp


def build_target_network(observation_size):

    network = Network(observation_size, hidden_size1, hidden_size2, output_size)
    for p in network.parameters():
        p.requires_grad_(False)

    return network


def build_predictor_network(observation_size):

    network = Network(observation_size, hidden_size1, hidden_size2, output_size)
    return network