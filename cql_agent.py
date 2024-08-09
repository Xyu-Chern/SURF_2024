from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn, optim

import numpy as np

import pytorch_util as ptu
from dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        qa_values = variables["qa_values"]
        q_values = variables["q_values"]

        batch_size = obs.size(0)


        repeated_obs = obs.unsqueeze(1).repeat(1, self.num_actions, 1).view(-1, obs.size(-1))
        all_action_q_values = self.critic(repeated_obs).view(batch_size, self.num_actions)

        logsumexp_q = torch.logsumexp(all_action_q_values / self.cql_temperature, dim=1)

        cql_loss = logsumexp_q.mean() - q_values.mean()

        loss = loss + self.cql_alpha * cql_loss

        metrics["cql_loss"] = cql_loss.item()

        return loss, metrics, variables



