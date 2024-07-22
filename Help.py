import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import minari
from minari import DataCollector
from typing import Union


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


def convert_features_to_tensor(state, next_state, action, intrinsic_reward):
    state_tensor = convert_tuple_to_tensor(state)
    next_state_tensor = convert_tuple_to_tensor(next_state)
    action_tensor = torch.tensor(action)
    i_reward_tensor = torch.tensor(intrinsic_reward)

