import torch
import numpy as np


def convert_dict_to_tensor(data_dict):
    # for key, value in data_dict.items():
    #     if not isinstance(value, (int, float, list, np.ndarray)):
    #         raise ValueError(f"Unsupported value type for key '{key}': {type(value)}")

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