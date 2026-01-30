import numpy as np
from typing import Tuple, Union, List, Dict, Optional, Callable, Any
import torch
import os
import random
import torch.nn as nn
from torch.distributions import Normal
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EXP_ADV_MAX = 100.0

TensorBatch = List[torch.Tensor]  # Define a pattern : TensorBatch

class Agent():
    def __init__(
        self,
        max_action: float,
        actor,
        actor_optimizer: torch.optim.Optimizer,
        q_network,
        q_optimizer: torch.optim.Optimizer,
        v_network,
        v_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        soft_update_lambda: float = 0.005,
        clip_epsilon: float = 0.2,
        update_timestep: int = 4000,
        K_epochs: int = 80,
        device: str = "cuda",
    ):
        self.device = device
        self.max_action = max_action
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.q_network = q_network
        self.q_optimizer = q_optimizer
        self.v_network = v_network
        self.v_optimizer = v_optimizer
        self.discount = discount
        self.soft_update_lambda = soft_update_lambda
        self.clip_epsilon = clip_epsilon
        self.update_timestep = update_timestep
        self.K_epochs = K_epochs
        self.memory = []

    def act(self, state):
        return self.actor.act(state, self.device)

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_returns(self, rewards, masks, values):
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.discount * values[step + 1] * masks[step] - values[step]
            gae = delta + self.discount * self.soft_update_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self):
        states = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)
        actions = torch.FloatTensor([t[1] for t in self.memory]).to(self.device)
        rewards = [t[2] for t in self.memory]
        masks = torch.FloatTensor([t[3] for t in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([t[4] for t in self.memory]).to(self.device)
        values = self.v_network(states)
        returns = self.compute_returns(rewards, masks, values)
        returns = torch.cat(returns).detach()

        advantages = returns - values.detach()

        for _ in range(self.K_epochs):
            action_mean, action_log_std = self.actor(states).sample()
            dist = Normal(action_mean, action_log_std.exp())
            log_probs = dist.log_prob(actions).sum(dim=-1)

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(returns, values)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.v_optimizer.zero_grad()
            critic_loss.backward()
            self.v_optimizer.step()

        self.memory = []



