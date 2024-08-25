
import numpy as np
from typing import Tuple ,Union, List, Dict ,Optional,Callable ,Any
import torch
TensorBatch = List[torch.Tensor] # Define a pattern : TesorBatch

import os
import random
import torch.nn as nn
from torch.distributions import Normal
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
EXP_ADV_MAX = 100.0


class Agent:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        memory , 
        algorithm_tau: float = 0.7, 
        beta: float = 2.0, 
        max_timesteps: int = 1000000,
        discount: float = 0.99, 
        soft_update_lamda: float = 0.005,
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.max_action = max_action
        self.q_network= q_network
        self.v_network = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_timesteps) 
        self.memory = memory
        # Help soft update
        # -- deepcopy : copy all the architechture and parameter
        # -- requires_grad_(False) : we add gradient in soft update
        self.q_target = copy.deepcopy(self.q_network).requires_grad_(False).to(device)

        self.algorithm_tau = algorithm_tau
        self.soft_update_lamda= soft_update_lamda
        self.beta = beta  # advantage weighted regression component
        self.total_it = 0 # interation steps

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        with torch.no_grad():
            q_target= self.q_target(observations, actions)
        adv_weights=q_target-self.v_network(observations)
        v_loss = asymmetric_l2_loss(adv_weights, self.algorithm_tau)
        log_dict["value_loss"] = v_loss.item() # get value without tensor
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv_weights

    def _update_q(self,next_v_value: torch.Tensor,observations: torch.Tensor,actions: torch.Tensor,rewards: torch.Tensor,terminals: torch.Tensor,log_dict: Dict,):
        #  next_v_value.detach() copy the tensor without gradient computation
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v_value.detach()
        double_q = self.q_network.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in double_q) / len(double_q)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.q_network, self.soft_update_lamda)

    def _update_policy(self,adv_weights: torch.Tensor,observations: torch.Tensor,actions: torch.Tensor,log_dict: Dict,):
        exp_adv_weights = torch.exp(self.beta * adv_weights.detach()).clamp(max=EXP_ADV_MAX) # clamp function provide a max value if e^ adv_weights over this value
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution): # isinstance(obj,class) : check if the object is from the class
            behaviour_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False) # get the log sum and (keepdim=False) remove the action dim
            # print("behaviour_losses : ", behaviour_losses)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            behaviour_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv_weights * behaviour_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step() # update actor network
        self.actor_lr_schedule.step() # update actor learning rate

    def train(self, batch_size):
        self.total_it += 1 
        batch_states, batch_actions, batch_rewards, batch_next_states , batch_dones = self.memory.sample(batch_size)
        batch_states = batch_states.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_next_states = batch_next_states.to(self.device)
        batch_dones = batch_dones.to(self.device)

        log_dict = {}
        with torch.no_grad():
            next_v_value = self.v_network(batch_next_states )
        adv = self._update_v(batch_states, batch_actions, log_dict)
        batch_rewards =  batch_rewards.squeeze(dim=-1) 
        self._update_q(next_v_value, batch_states,  batch_actions,  batch_rewards, batch_dones, log_dict)
        self._update_policy(adv, batch_states, batch_actions, log_dict)
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_network": self.v_network.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.q_network.load_state_dict(state_dict["q_network"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.q_network)

        self.v_network.load_state_dict(state_dict["v_network"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"] 

    def act(self , state):
        action = self.actor.act(state , self.device)
        return action


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    # torch.abs()  absolute value
    # torch.mean() calculate the expectation
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def soft_update(target: nn.Module, source: nn.Module, lamda: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            (1 - lamda) * target_param.data + lamda * source_param.data
            )
