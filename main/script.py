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

import argparse

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Run the RL agent with different hyperparameters.")
parser.add_argument('--project', type=str, default='Surf_2024_v2', help="Name of the project")
parser.add_argument('--name', type=str, default='hyperparameter', help="Name of the run")
parser.add_argument('--env', type=str, default="PointMaze_UMazeDense-v3", help="Environment name")
parser.add_argument('--path', type=str, default="PointMaze_UMazeDense-v3_PPO&RND_v2", help="Path")
parser.add_argument('--k', type=int, default=16, help="K value")
parser.add_argument('--gamma', type=float, default=0.99, help="Gamma value")
parser.add_argument('--v_lr', type=float, default=0.001, help="Value function learning rate")
parser.add_argument('--q_lr', type=float, default=0.001, help="Q function learning rate")
parser.add_argument('--a_lr', type=float, default=0.001, help="Actor learning rate")
parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
parser.add_argument('--test_frequency', type=int, default=int(5e2), help="Frequency of testing")
parser.add_argument('--test_epochs', type=int, default=100, help="Number of test epochs")
parser.add_argument('--test_timesteps', type=int, default=100, help="Number of test timesteps")
parser.add_argument('--run_time', type=int, default=int(1e6), help="Total runtime steps")
parser.add_argument('--env_render',type =bool , default= False, help="Render the environment during training")
parser.add_argument('--tau',type =int , default= 0.7, help="iql tau") # range (0,1)
parser.add_argument('--beta',type =int , default= 2.0, help="iql beta")

args = parser.parse_args()

# Initialize wandb
wandb.init(project=args.project)
wandb.watch_called = False  
config = wandb.config  
config.name = f"{args.name}"
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.path_list = ["PointMaze_UMazeDense-v3_PPO&RND_v0", 
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
               "PointMaze_UMazeDense-v3_RND_v2"]
config.path = args.path
config.env = args.env
config.k = args.k
config.gamma = args.gamma
config.v_lr = args.v_lr
config.q_lr = args.q_lr
config.a_lr = args.a_lr
config.batch_size = args.batch_size
config.test_frequency = args.test_frequency
config.test_epochs = args.test_epochs
config.test_timesteps = args.test_timesteps
config.run_time = args.run_time
config.env_render = args.env_render
config.tau = args.tau
config.beta = args.beta
wandb.run.name = f"{config.name}"
wandb.run.save()

replay_buffer = ReplayBuffer(config.path)
memory = replay_buffer.get_memory(k=config.k)

# Set environment
if config.env_render:
    env = gym.make(config.env, render_mode="human")
else:
    env = gym.make(config.env)

# Prepare and set agent
observation_dim = env.observation_space['observation'].shape[0]
achieved_goal_dim = env.observation_space['achieved_goal'].shape[0]
state_dim = observation_dim + achieved_goal_dim
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

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
              memory=memory,
              discount = config.gamma,
              algorithm_tau= config.tau,
              beta=config.beta,
              device=config.device)

def obsList_to_state_interaction(obsList):
    state = np.hstack([obsList['observation'], obsList['desired_goal']])
    return state

# Evaluation function
def eval(env, agent, epochs=config.test_epochs, timsteps=config.test_timesteps):
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
        episode_reward_list.append(episode_reward / timsteps)

    agent.actor.train()
    score = torch.tensor(episode_reward_list).mean()
    return score

# Main training loop
def main():
    for step in range(config.run_time):
        log_dict = agent.train(config.batch_size)
        if step % config.test_frequency == 0:
            score = eval(env, agent)
            print(f"Step {step}, Score: {score.item()}")

            log_dict["Score"] = score.item() 
            wandb.log(log_dict)
            torch.save(agent.state_dict(), f'./results/scripts/{config.name}.pt')

if __name__ == "__main__":
    main()




