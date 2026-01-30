
import torch
import numpy as np
import minari
from numpy.random import default_rng

rng = default_rng()

import torch
import numpy as np
import minari
from numpy.random import default_rng

rng = default_rng()

class ReplayBuffer:
    def __init__(self, path: str, seed=42, device: str = "cpu", limit_to_10=False):
        self._device = device
        self.seed = seed
        self.limit_to_10 = limit_to_10  
        self.normal_transitions, self.hindsight_transitions = self._get_transitions(minari.load_dataset(path))   

    def __len__(self):
        return self.size
    
    def _get_transitions(self, dataset):
        if self.limit_to_10:
            self.size = min(10, dataset.total_episodes)  #
        else:
            self.size = dataset.total_episodes  
            
        normal_transitions = []
        hindsight_transitions = []
        for idx, episode_data in enumerate(dataset.iterate_episodes()):
            if self.limit_to_10 and idx >= 10:  
                break
            observations, achieved_goals, desired_goals, actions, rewards, terminations = self._get_trajactory_list(episode_data)
            for i in range(len(actions)):
                normal_transitions.append(self._get_normal_transitions(observations, achieved_goals, desired_goals, actions, rewards, terminations, i))
                hindsight_transitions.append(self._get_multistep_transitions(observations, achieved_goals, actions, rewards, terminations, i))
        return normal_transitions, hindsight_transitions

    def _get_trajactory_list(self , episode_data):
        observations = episode_data.observations['observation']
        achieved_goals = episode_data.observations['achieved_goal']
        desired_goals = episode_data.observations['desired_goal']
        actions = episode_data.actions
        rewards = episode_data.rewards
        terminations = episode_data.terminations.astype(int)
        return  observations , achieved_goals , desired_goals, actions , rewards , terminations
    
    def _get_normal_transitions(self , observations, achieved_goals,  desired_goals , actions, rewards, terminations, i):
        
        state = np.hstack([observations[i] , desired_goals[i]])
        next_state  = np.hstack([observations[i+1] , desired_goals[i]])

        distance= np.linalg.norm( achieved_goals[i+1] - desired_goals[i] , axis=-1)
        reward = np.exp(-distance)
        # assert np.abs(reward - rewards[i]) < 1e-3, (reward, observations[i], achieved_goals[i], observations[i+1],achieved_goals[i+1],  desired_goals[i] , actions[i], rewards[i])
        return (
            self._to_tensor(state), 
            self._to_tensor(actions[i]), 
            self._to_tensor(reward), 
            self._to_tensor(next_state) ,
            self._to_tensor(terminations[i]),
        )
               
    def _get_multistep_transitions(self , observations , achieved_goals , actions , rewards , terminations , i ):
        transitions = []
        for j in range(i, len(actions)+1):
            desired_goals = achieved_goals[j]
            state = np.hstack([observations[i] , desired_goals])
            next_state  = np.hstack([observations[i+1] , desired_goals])
            distance= np.linalg.norm(achieved_goals[i+1] - desired_goals, axis=-1)
            reward = np.exp(-distance)
            transitions.append((
                self._to_tensor(state), 
                self._to_tensor(actions[i]), 
                self._to_tensor(reward), 
                self._to_tensor(next_state) ,
                self._to_tensor(0),
            ))
        return transitions
        
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
    
    def get_memory(self, k =16, keep_normal=True):
        return Memory(self.normal_transitions, self.hindsight_transitions, k, keep_normal)


class Memory():
    def __init__(self, normal_transitions, hindsight_transitions, k, keep_normal=True):
        self.normal_transitions = normal_transitions
        self.hindsight_transitions = hindsight_transitions
        self.num_episodes = len(normal_transitions) 
        # print ("num_transitions",len(normal_transitions))
        self.k = k
        self.keep_normal = keep_normal
        self.start = 0
        self.shuffle()

    def shuffle(self):
        self.index = rng.choice(self.num_episodes, size=self.num_episodes, replace=False)

    def sample(self, batch_size):
        #batch size 512 / 16 = 32 
        np.random.seed() # if set the self.seed it will output solid data
        
        replace = batch_size > len(self.normal_transitions)
        indices = [self.index[ (self.start+i ) % self.num_episodes] for i in range(batch_size)]

        self.start += batch_size
        if self.start > self.num_episodes:
            self.shuffle()
            self.start = self.start % self.num_episodes

        if self.keep_normal:
            states = [self.normal_transitions[i][0] for i in indices]
            actions = [self.normal_transitions[i][1] for i in indices]
            rewards = [self.normal_transitions[i][2] for i in indices]
            next_states = [self.normal_transitions[i][3] for i in indices]
            dones = [self.normal_transitions[i][4] for i in indices]   
        else:
            states, actions , rewards, next_states, dones = [],[],[],[],[]
        
        if self.k > 0:
            for i in indices:
                hindsight_transitions = self.hindsight_transitions[i][:self.k]
                if len(self.hindsight_transitions[i]) > 0:
                    replace = self.k > len(hindsight_transitions)
                    # future_indices = np.random.randint(0, len(self.hindsight_transitions[i]), size=self.k)
                    # size = min(len(hindsight_transitions),self.k )
                    future_indices = rng.choice(len(hindsight_transitions), size = self.k, replace = replace)
                    states += [hindsight_transitions[j][0] for j in future_indices]
                    actions += [hindsight_transitions[j][1] for j in future_indices]
                    rewards += [hindsight_transitions[j][2] for j in future_indices]
                    next_states += [hindsight_transitions[j][3] for j in future_indices]
                    dones += [hindsight_transitions[j][4] for j in future_indices]

        batch_states = torch.stack(states)
        batch_actions  = torch.stack(actions)
        batch_rewards= torch.stack(rewards)
        batch_next_states = torch.stack(next_states)
        batch_dones  = torch.stack(dones)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
if __name__ == "__main__":
    memory = ReplayBuffer(path = "PointMaze_UMazeDense-v3_PPO_5000" , limit_to_10= True).get_memory()
    print(memory.num_episodes)
    list = [1, 2]
    print(list[:16])
    s = "5000_v"
    if True:
        s = s.replace("5000", "1000")
        print(s)