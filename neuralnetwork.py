import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ConvClass(nn.Module):
    def __init__(self):    
        super().__init__()#NeuralNetwork, self
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, convolutions):    
        super().__init__()#NeuralNetwork, self
        self.convs = convolutions
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)        

    def forward(self, x):
        x = self.convs(x)        
        x = x.view(x.shape[0], -1) #refit x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuralNetwork2(nn.Module):
    def __init__(self, convolutions):    
        super().__init__()#NeuralNetwork, self
        self.convs = convolutions
        self.fc1 = nn.Linear(256+11, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 11)
        # uniform init layer 3

    def forward(self, state, action):
        x = self.convs(x)        
        x = x.view(x.shape[0], -1) #refit x
        x = F.relu(self.fc1(torch.cat(state, action, dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuralNetwork3(nn.Module):
    def __init__(self, convolutions):    
        super().__init__()#NeuralNetwork, self
        self.convs = convolutions
        self.fc1 = nn.Linear(256+11, 128)
        self.fc2 = nn.Linear(128, 64)

        self.mean_fc = nn.Linear(64, 11)
        # mean, init uniform
        self.log_variance_fc = nn.Linear(64, 11)
        # variance, init uniform
        # mean, variance, normal distribution

    def forward(self, state):
        x = self.convs(x)        
        x = x.view(x.shape[0], -1) #refit x
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_variance = self.log_variance_fc(x)
        log_variance = torch.clamp(log_variance, -20, 2)
        return mean, log_variance

    def sample(self, state, epsilon=1e-6):
        mean, log_variance = self.forward(state)
        variance = log_variance.exp()
        gaussian = Normal(mean, log_variance)        
        z = gaussian.rsample()
        
        log_pi = (gaussian.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)).sum(1, keepdim=True)
        return mean, variance, z, log_pi