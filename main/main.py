import os
os.add_dll_directory("C://Users//28449//.mujoco//mjpro150//bin")
import sys
sys.path.append('.')

import gymnasium as gym
import torch
import wandb
import numpy as np

from agent.model_utils import QFunction, ValueFunction, GaussianPolicy
from utils.replaybuffer import ReplayBuffer
from agent.iql import Agent

# Initialize wandb
wandb.init(project='Surf_2024')
wandb.watch_called = False
config = wandb.config
config.name = "PointMaze_iql_36type"
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.paths = [
    "PointMaze_UMazeDense-v3_PPO&RND_v0",
    "PointMaze_UMazeDense-v3_PPO&RND_v1",
    "PointMaze_UMazeDense-v3_PPO&RND_v2",
    "PointMaze_UMazeDense-v3_PPO_v0",
    "PointMaze_UMazeDense-v3_PPO_v1",
    "PointMaze_UMazeDense-v3_PPO_v2",
    "PointMaze_UMazeDense-v3_random_v0",
    "PointMaze_UMazeDense-v3_random_v1",
    "PointMaze_UMazeDense-v3_random_v2",
    "PointMaze_UMazeDense-v3_RND_v0",
    "PointMaze_UMazeDense-v3_RND_v1",
    "PointMaze_UMazeDense-v3_RND_v2"
]
config.env = "PointMaze_UMazeDense-v3"
config.gamma = 0.99
config.v_lr = 0.001
config.q_lr = 0.001
config.a_lr = 0.001
config.batch_size = 512
config.test_frequency = int(1e3)
config.test_epochs = 10
config.test_timesteps = 1000
config.run_time = int(1e6)
config.env_render = False
wandb.run.name = f"{config.name}"
wandb.run.save()

# Set up ReplayBuffers and Agents
replay_buffers = [ReplayBuffer(path) for path in config.paths]
memories = [[buffer.get_memory(k = i) for i in [0, 1, 4]] for buffer in replay_buffers]

# set env
if config.env_render:
    env = gym.make(config.env, render_mode="human")
else:
    env = gym.make(config.env)

observation_dim = env.observation_space['observation'].shape[0]
achieved_goal_dim = env.observation_space['achieved_goal'].shape[0]
state_dim = observation_dim + achieved_goal_dim 
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# set agent list
agents = []
for i in range(len(config.paths)):
    for j in range(3):
        q_network = QFunction(state_dim, action_dim).to(config.device)
        v_network = ValueFunction(state_dim).to(config.device)
        actor = GaussianPolicy(state_dim, action_dim, max_action).to(config.device)
        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.v_lr)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.q_lr)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.a_lr)
        
        agent = Agent(max_action,
                      actor,
                      actor_optimizer,
                      q_network,
                      q_optimizer,
                      v_network,
                      v_optimizer,
                      memory=memories[i][j],
                      device=config.device)
        agents.append(agent)



# change_state for agent
def obsList_to_state_interaction(obsList):
    return np.hstack([obsList['observation'], obsList['desired_goal']])

# eval
def eval(env, agent, epochs=10, timsteps=100):
    agent.actor.eval()
    episode_reward_list = []
    for ep in range(epochs):
        obsList, info = env.reset()
        state = obsList_to_state_interaction(obsList)
        episode_reward = 0.0
        for t in range(timsteps):
            action = agent.act(state)
            obsList, reward, done, truncated, info = env.step(action)
            state = obsList_to_state_interaction(obsList)
            episode_reward += reward
        episode_reward_list.append(episode_reward/timsteps)
        
    agent.actor.train()
    score = torch.tensor(episode_reward_list).mean()
    return score

# main
for step in range(config.run_time):
    log_dicts = [agent.train(config.batch_size) for agent in agents]

    wandb_log_data = {}
    for i, log_dict in enumerate(log_dicts):
        wandb_log_data[f"Value Loss_{i}"] = log_dict['value_loss']
        wandb_log_data[f"Q Loss_{i}"] = log_dict['q_loss']
        wandb_log_data[f"Actor Loss_{i}"] = log_dict['actor_loss']

    wandb.log(wandb_log_data)

    if step % config.test_frequency == 0:
        scores = [eval(env, agent) for agent in agents]

        wandb_log_data = {f"Score_{i}": score.item() for i, score in enumerate(scores)}
        wandb.log(wandb_log_data)

        for i, agent in enumerate(agents):
            torch.save(agent.state_dict(), f'./results/{config.name}_d{i}.pt')

        print(f"Step {step}, Scores: {[score.item() for score in scores]}")
