import torch
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, batch_size):
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = batch_size

    def sample(self):        
        samples = random.sample(self.replay_buffer, k=self.batch_size)     
        states = torch.tensor([s[0] for s in samples]).float().to(self.device)        
        actions = torch.tensor([s[1] for s in samples]).float().to(self.device)
        rewards = torch.tensor([s[2] for s in samples]).float().unsqueeze(1).to(self.device)
        next_states = torch.tensor([s[3] for s in samples]).float().to(self.device)
        dones = torch.tensor([s[4] for s in samples]).float().unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones