import torch
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, batch_size):
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = batch_size
        self.device = torch.device("cpu")

    def sample(self):        
        samples = random.sample(self.replay_buffer, k=self.batch_size)    
         
        states = torch.stack([s[0] for s in samples])
        actions = torch.tensor([s[1] for s in samples]).float().to(self.device)
        rewards = torch.tensor([s[2] for s in samples]).float().unsqueeze(1).to(self.device)
        next_states = torch.stack([s[3] for s in samples])
        dones = torch.tensor([s[4] for s in samples]).float().unsqueeze(1).to(self.device)
        print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        return states, actions, rewards, next_states, dones
    
    def add(self, sars):
        self.replay_buffer.append(sars)