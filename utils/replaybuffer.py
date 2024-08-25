
import torch
import numpy as np
import minari

class ReplayBuffer:
    def __init__(self, path: str, list_k = [0, 4, 16], modify_reward = True, seed = 42, device: str = "cpu"):
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
        transitions = []
        if k == 0:
            for episode_data in dataset.iterate_episodes():
                observations , achieved_goals , actions , rewards , terminations = self._get_trajactory_list(episode_data)
                for i in range(len(actions)):
                    self._add_normal_transitions(transitions, observations, achieved_goals, actions, rewards, terminations, i)

        if k == 1 : 
            for episode_data in dataset.iterate_episodes():
                observations , achieved_goals , actions , rewards , terminations = self._get_trajactory_list(episode_data)
                for i in range(len(actions)):
                    self._add_normal_transitions(transitions, observations, achieved_goals, actions, rewards, terminations, i)
                    self._add_onestep_transitions(transitions, observations, achieved_goals, actions, rewards, terminations, i)

        if k > 1 : 
            for episode_data in dataset.iterate_episodes():
                observations , achieved_goals , actions , rewards , terminations = self._get_trajactory_list(episode_data)
                for i in range(len(actions)):
                    self._add_normal_transitions(transitions, observations, achieved_goals, actions, rewards, terminations, i)
                    self._add_multistep_transitions(transitions, observations , achieved_goals , actions , rewards , terminations, i, k)

        return (transitions, len(transitions))
    
    def _get_trajactory_list(self , episode_data):
        observations = episode_data.observations['observation']
        achieved_goals = episode_data.observations['achieved_goal']
        desired_goals = episode_data.observations['desired_goal']
        actions = episode_data.actions
        rewards = episode_data.rewards
        terminations = episode_data.terminations.astype(int)
        return  observations , achieved_goals , actions , rewards , terminations
    
    def _add_normal_transitions(self , transitions, observations, achieved_goals, actions, rewards, terminations, i):
        state = np.hstack([observations[i] , achieved_goals[i]])
        next_state  = np.hstack([observations[i+1] , achieved_goals[i]])
        if self.modify_reward: 
            distance= np.linalg.norm(achieved_goals[i] - observations[i+1][:2], axis=-1)
            reward = np.exp(-distance)
        else:
            reward = rewards[i]
        transitions.append((
            self._to_tensor(state), 
            self._to_tensor(actions[i]), 
            self._to_tensor(reward), 
            self._to_tensor(next_state) ,
            self._to_tensor(terminations[i]),
        ))

    def _add_onestep_transitions(self , transitions, observations , achieved_goals , actions , rewards , terminations , i):
        if i + 1 < len(actions):
            goal =  achieved_goals[i+1]
        else :
            goal =  achieved_goals[i]

        state = np.hstack( [observations[i] , goal])
        next_state  = np.hstack([observations[i+1] , goal])
        if self.modify_reward: 
            distance= np.linalg.norm(goal - observations[i+1][:2], axis=-1)
            reward = np.exp(-distance)
        else:
            reward = rewards[i]
        transitions.append((
            self._to_tensor(state), 
            self._to_tensor(actions[i]), 
            self._to_tensor(reward), 
            self._to_tensor(next_state) ,
            self._to_tensor(terminations[i]),
        ))

    def _add_multistep_transitions(self , transitions, observations , achieved_goals , actions , rewards , terminations , i , k):
        np.random.seed(self.seed)
        indices = np.random.randint(i, len(actions), size=k)
        for j in indices :
            goal = achieved_goals[j]
            state = np.hstack([observations[i] , goal])
            next_state  = np.hstack([observations[i+1] , goal])
            if self.modify_reward: 
                distance= np.linalg.norm(goal - observations[i+1][:2], axis=-1)
                reward = np.exp(-distance)
            else:
                reward = rewards[i]
            transitions.append((
                self._to_tensor(state), 
                self._to_tensor(actions[i]), 
                self._to_tensor(reward), 
                self._to_tensor(next_state) ,
                self._to_tensor(terminations[i]),
            ))

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
    
    def get_memory(self , k):
        transitions, max_batch_size = self.datasets_map[k]
        return Memory(transitions, max_batch_size)


class Memory():
    def __init__(self, transitions, max_batch_size ):
        self.transitions = transitions
        self.max_batch_size = max_batch_size 

    def sample(self, batch_size):

        np.random.seed() # if set the self.seed it will output solid data
        indices = np.random.randint(0, self.max_batch_size, size=batch_size)

        batch_states = torch.stack([self.transitions[i][0] for i in indices])
        batch_actions  = torch.stack([self.transitions[i][1] for i in indices])
        batch_rewards= torch.stack([self.transitions[i][2] for i in indices])
        batch_next_states = torch.stack([self.transitions[i][3] for i in indices])
        batch_dones  = torch.stack([self.transitions[i][4] for i in indices])

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
  
if __name__ == "__main__":
    replay_buffer = ReplayBuffer("PointMaze_UMazeDense-v3_PPO_v0")
    memory0 = replay_buffer.get_memory(k = 0)
    memory1 = replay_buffer.get_memory(k = 1)
    memory4 = replay_buffer.get_memory(k = 4)
    
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = memory0.sample(32)
    print(batch_states.shape , batch_actions.shape ,batch_rewards.shape ,batch_next_states.shape, batch_dones.shape)

    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = memory1.sample(32)
    print(batch_states.shape , batch_actions.shape ,batch_rewards.shape ,batch_next_states.shape, batch_dones.shape)

    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = memory4.sample(32)
    print(batch_states.shape , batch_actions.shape ,batch_rewards.shape ,batch_next_states.shape, batch_dones.shape)





