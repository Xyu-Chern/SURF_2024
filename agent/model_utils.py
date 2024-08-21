
import numpy as np
from typing import Tuple ,Optional,Callable
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

class QFunction(nn.Module):
    def __init__(self, state_dimension: int, action_dimension: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        # dims=[state_dimension + action_dimension,hidden_dim,.....,hidden_dim,1] There are n_hidden numbers of hidden_dim
        dims = [state_dimension + action_dimension, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action)) # trick : mitigate overestimation problem
    
    # combine state with action
    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_a = torch.cat([state, action], 1)
        return self.q1(s_a), self.q2(s_a)

class MLP(nn.Module):
    def __init__(
        self, 
        dims ,
        # Define activation_fn, with 0 arguments and return a nn.Module object, default is nn.ReLU
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None, # design arg use dropout = 0.1 below
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")
        
        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())

        # The squeeze operation is often used to remove dimensions of size 1 from a tensor to simplify the shape of the tensor.
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        # * is an unpacking operator that unpacks list, tuple, etc and passes its elements as separate arguments to a function 
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class ValueFunction(nn.Module):
    def __init__(self, state_dimension: int, hidden_dimension: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dimension, *([hidden_dimension] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)
     
class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP([state_dimension, *([hidden_dim] * n_hidden), action_dimension],output_activation_fn=nn.Tanh,dropout=dropout)
        # What use of log_std ?
        self.log_std = nn.Parameter(torch.zeros(action_dimension, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        # self.training is in torch.nn.Module 
        # when we use model.train(), self.training = True
        # when we use model.eval(), self.training = False
        if not self.training :
            action = self.forward(state).mean 
        else:
            action = self.forward(state).sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()