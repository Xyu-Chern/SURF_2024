import sys
sys.path.append('.')
import os
os.add_dll_directory("C://Users//28449//.mujoco//mjpro150//bin")

import gymnasium as gym
import torch
import numpy as np

from agent.model_utils import QFunction, ValueFunction, GaussianPolicy
from agent.ppo import Agent

import wandb
wandb.init(project="testppo")
wandb.watch_called = False  
wandb.run.name = "ppo - v1"
wandb.run.save()

env = gym.make("PointMaze_UMazeDense-v3")

observation_dim = env.observation_space['observation'].shape[0]
achieved_goal_dim = env.observation_space['achieved_goal'].shape[0]
state_dim = observation_dim + achieved_goal_dim
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
device = "cuda" if torch.cuda.is_available() else "cpu"

q_network = QFunction(state_dim, action_dim, n_hidden=2, dropout=0.2).to(device)
v_network = ValueFunction(state_dim, n_hidden=2, dropout=0.2).to(device)
actor = GaussianPolicy(state_dim, action_dim, max_action, n_hidden=2, dropout=0.2).to(device)

v_optimizer = torch.optim.Adam(v_network.parameters(), lr=0.0001, weight_decay=1e-5)
q_optimizer = torch.optim.Adam(q_network.parameters(), lr=0.0001, weight_decay=1e-5)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001, weight_decay=1e-5)

agent = Agent(
    max_action=max_action,
    actor=actor,
    actor_optimizer=actor_optimizer,
    q_network=q_network,
    q_optimizer=q_optimizer,
    v_network=v_network,
    v_optimizer=v_optimizer,
    device=device
)

def obsList_to_state_interaction(obsList):
    state = np.hstack([obsList['observation'], obsList['desired_goal']])
    return state

def eval(env, agent, epochs=100, timsteps=100):
    episode_reward_list = []
    for ep in range(epochs):
        obsList, info = env.reset()
        state = obsList_to_state_interaction(obsList)
        episode_reward = 0.0
        for t in range(timsteps):
            action, action_log_prob = agent.act(state)
            obsList, reward, done, truncated, info = env.step(action)
            state = obsList_to_state_interaction(obsList)
            episode_reward += reward
            if done: 
                episode_reward += (timsteps- t ) * reward
                break
        episode_reward_list.append(episode_reward / timsteps)

    score = torch.tensor(episode_reward_list).mean()
    return score, torch.tensor(episode_reward_list).std()

if __name__ == "__main__":
    run_time = 300000
    timsteps=100

    for i in range(run_time):
        state = env.reset()
        done = False
        for t in range(timsteps):
            while not done:
                action, action_log_prob = agent.act(state)
                obsList, reward, done, truncated, info = env.step(action)
                state = obsList_to_state_interaction(obsList)
                agent.store_transition((state, action, reward, 1 - int(done), action_log_prob))
        if i % agent.update_timestep == 0:
            agent.update()
            score , std = eval(env,agent)
            wandb.log({"score " : score.item()})


