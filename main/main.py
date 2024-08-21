import os
os.add_dll_directory("C://Users//28449//.mujoco//mjpro150//bin")
import sys
sys.path.append('.')

import gymnasium as gym
import torch
import wandb

from agent.model_utils import QFunction , ValueFunction , GaussianPolicy
from utils.replaybuffer import ReplayBuffer
from agent.iql import Agent

# Initialize wandb
wandb.init(project='Surf_2024')
wandb.watch_called = False  
config = wandb.config  
config.name = "test_iql_rndversion"
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.dataset_path = "PointMaze_UMazeDense-v3_v1"
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
wandb.run.name = f"{config.name}"
wandb.run.save()

# set datasets and env 
memory = ReplayBuffer(config.dataset_path)
env = gym.make(config.env ,  render_mode="human")

observation_dim = env.observation_space['observation'].shape[0]
achieved_goal_dim = env.observation_space['achieved_goal'].shape[0]
desired_goal_dim = env.observation_space['desired_goal'].shape[0]
state_dim = observation_dim + achieved_goal_dim + desired_goal_dim
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# prepare and set agent
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
        memory = memory,
        device = config.device 
    )

# eval
def eval(env, agent, epochs = 10 , timsteps = 100):  
    agent.actor.eval()
    episode_reward_list = []
    for ep in range(epochs): 
        state , info = env.reset()
        episode_reward = 0.0
        for t in range(timsteps):  
            action = agent.act(state)
            state, reward, done, truncated, info = env.step(action)
            env.render()
            episode_reward += reward
        episode_reward_list.append(episode_reward/timsteps)
        
    agent.actor.train() 
    score = torch.tensor(episode_reward_list).mean() 
    # score=  (score - min_data_reward / max_data_reward - min_data_reward)* 100
    return score

# main
for step in range(config.run_time):
    log_dict = agent.train(config.batch_size)
    wandb.log(log_dict)

    if step % config.test_frequency == 0:
        score = eval(env, agent)
        wandb.log({"Testing Score": score.item()})
        print(f"Step {step}, Score: {score.item()}")
        torch.save(agent.state_dict(), f'./results/{config.name}_{config.dataset_path}.pt')


    

