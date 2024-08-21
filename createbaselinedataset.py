import gymnasium as gym
from stable_baselines3.ppo.ppo_baseline import PPO


import minari
from minari import DataCollector


env = DataCollector(gym.make("PointMaze_UMazeDense-v3", render_mode="rgb_array"))
# env = gym.make("PointMaze_UMazeDense-v3", render_mode="rgb_array")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

dataset = env.create_dataset(dataset_id="PointMaze_UMazeDense-v3_PPO_v0",
                             eval_env="PointMaze_UMazeDense-v3",
                             minari_version="3",
                             algorithm_name="ppo")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)