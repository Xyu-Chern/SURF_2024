import os
import torch
import numpy as np
import minari

os.add_dll_directory("C://Users//28449//.mujoco//mjpro150//bin")

class ReplayBuffer:
    def __init__(self, path: str, list_k = [0, 1, 4], modify_reward = False, seed = 42, device: str = "cpu"):
        self._device = device
        self.seed =seed
        self.list_k = list_k
        self.modify_reward = modify_reward 
        self.datasets_map = self._load_dataset(path)

    def _load_dataset(self, path):
        dataset = minari.load_dataset(path)
        dataset_map = {}
        for k in self.list_k:
            dataset_map[k] = self._set_single_transitions(dataset, k)
        return dataset_map

    def _set_single_transitions(self, dataset, k):
        
        if k == 0:
            transitions = []
            for episode_data in dataset.iterate_episodes():

                observations = episode_data.observations['observation']
                achieved_goals = episode_data.observations['achieved_goal']
                desired_goals = episode_data.observations['desired_goal']
                actions = episode_data.actions
                rewards = episode_data.rewards
                terminations = episode_data.terminations.astype(int)

                for i in range(len(actions)):
                    transitions.append((
                        self._to_tensor(observations[i]), 
                        self._to_tensor(achieved_goals[i]), 
                        self._to_tensor(actions[i]), 
                        self._to_tensor(rewards[i]), 
                        self._to_tensor(observations[i + 1]) ,
                        self._to_tensor(terminations[i]),
                    ))
        if k == 1 : 
            transitions = []
            for episode_data in dataset.iterate_episodes():

                observations = episode_data.observations['observation']
                achieved_goals = episode_data.observations['achieved_goal']
                desired_goals = episode_data.observations['desired_goal']
                actions = episode_data.actions
                rewards = episode_data.rewards
                terminations = episode_data.terminations.astype(int)

                for i in range(len(actions)):
                    transitions.append((
                        self._to_tensor(observations[i]), 
                        self._to_tensor(achieved_goals[i]), 
                        self._to_tensor(actions[i]), 
                        self._to_tensor(rewards[i]), 
                        self._to_tensor(observations[i + 1]) ,
                        self._to_tensor(terminations[i]),
                    ))
                    transitions.append((
                        self._to_tensor(observations[i]), 
                        self._to_tensor(achieved_goals[i+1]) if i + 1 < len(actions) else self._to_tensor(achieved_goals[i]), 
                        self._to_tensor(actions[i]), 
                        self._to_tensor(rewards[i]), 
                        self._to_tensor(observations[i + 1]) ,
                        self._to_tensor(terminations[i]),
                    ))
        if k > 1 : 
            transitions = []

            for episode_data in dataset.iterate_episodes():

                observations = episode_data.observations['observation']
                achieved_goals = episode_data.observations['achieved_goal']
                desired_goals = episode_data.observations['desired_goal']
                actions = episode_data.actions
                rewards = episode_data.rewards
                terminations = episode_data.terminations.astype(int)
                
                for i in range(len(actions)):
                    np.random.seed(self.seed)
                    indices = np.random.randint(i, len(actions), size=k)
                    transitions.append((
                            self._to_tensor(observations[i]), 
                            self._to_tensor(achieved_goals[i]), 
                            self._to_tensor(actions[i]), 
                            self._to_tensor(rewards[i]), 
                            self._to_tensor(observations[i + 1]) ,
                            self._to_tensor(terminations[i]),
                        ))
                    for j in indices :
                        transitions.append((
                            self._to_tensor(observations[i]), 
                            self._to_tensor(achieved_goals[j]), 
                            self._to_tensor(actions[i]), 
                            self._to_tensor(rewards[i]), 
                            self._to_tensor(observations[i + 1]) ,
                            self._to_tensor(terminations[i]),
                        ))
        return (transitions, len(transitions))

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
    
    def get_data(self , k):
        transitions, max_batch_size = self.datasets_map[k]
        return transitions, max_batch_size

    def sample_transitions(self, k , batch_size: int):
        transitions, max_batch_size = self.datasets_map[k]
        np.random.seed() # if set the self.seed it will output solid data
        indices = np.random.randint(0, max_batch_size, size=batch_size)

        batch_obs = torch.stack([transitions[i][0] for i in indices])
        batch_achieved_goals = torch.stack([transitions[i][1] for i in indices])
        batch_actions = torch.stack([transitions[i][2] for i in indices])
        batch_rewards = torch.stack([transitions[i][3] for i in indices])
        batch_next_obs = torch.stack([transitions[i][4] for i in indices])
        batch_dones = torch.stack([transitions[i][5] for i in indices])

        return batch_obs, batch_achieved_goals, batch_actions, batch_rewards, batch_next_obs, batch_dones

if __name__ == "__main__":
    memory = ReplayBuffer("PointMaze_UMazeDense-v3_PPO_v0")
    batch_obs, batch_achieved_goals, batch_actions, batch_rewards, batch_next_obs, batch_dones = memory.sample_transitions(k = 4, batch_size = 128)
    print(batch_achieved_goals)





