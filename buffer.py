import torch
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, batch_size, device):
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = batch_size
        self.device = device

    def sample(self):        
        samples = random.sample(self.replay_buffer, k=self.batch_size)         
        states = torch.stack([s[0] for s in samples]).to(self.device)
        actions = torch.stack([torch.tensor(s[1]) for s in samples]).long().to(self.device)
        rewards = torch.tensor([s[2] for s in samples]).float().unsqueeze(1).to(self.device)
        next_states = torch.stack([s[3] for s in samples]).to(self.device)
        dones = torch.tensor([s[4] for s in samples]).float().unsqueeze(1).to(self.device)
        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        return states, actions, rewards, next_states, dones
    
    def add(self, sars):
        self.replay_buffer.append(sars)