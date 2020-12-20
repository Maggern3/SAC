import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

convoutputsize = 1296 
initial_weight = 3e-3

class ConvNetwork(nn.Module):
    def __init__(self):    
        super().__init__()#NeuralNetwork, self
        # 84x84x3
        self.conv1 = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        # 42x42x16, pool 21x21x16
        self.conv2 = nn.Conv2d(4, 16, 3, stride=1, padding=0)
        # 19x19x32, pool 9x9x16
        #self.conv3 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, state):
        x = self.pool(F.relu(self.conv1(state)))        
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(x.shape[0], -1) #refit x
        return x

class StateValueNetwork(nn.Module):
    def __init__(self):    
        super().__init__()#NeuralNetwork, self
        self.conv_net = ConvNetwork()#.to(self.device)
        self.fc1 = nn.Linear(convoutputsize, 256)
        self.fc2 = nn.Linear(256, 1)
        #self.fc3 = nn.Linear(64, 1)        

    def forward(self, state):
        x = self.conv_net(state)        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc3(x)
        return x

class ActionValueNetwork(nn.Module):
    def __init__(self):    
        super().__init__()#NeuralNetwork, self
        self.conv_net = ConvNetwork()#.to(self.device)
        self.fc1 = nn.Linear(convoutputsize+11, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # uniform init layer 3
        self.fc3.weight.data.uniform_(-initial_weight, initial_weight)
        self.fc3.bias.data.uniform_(-initial_weight, initial_weight)

    def forward(self, state, action):
        x = self.conv_net(state)
        x = F.relu(self.fc1(torch.cat((x, action), dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self):    
        super().__init__()#NeuralNetwork, self
        self.conv_net = ConvNetwork()#.to(self.device)
        self.fc1 = nn.Linear(convoutputsize, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mean_fc = nn.Linear(256, 11)
        self.mean_fc.weight.data.uniform_(-initial_weight, initial_weight)
        self.mean_fc.bias.data.uniform_(-initial_weight, initial_weight)
        self.log_variance_fc = nn.Linear(256, 11)
        self.log_variance_fc.weight.data.uniform_(-initial_weight, initial_weight)
        self.log_variance_fc.bias.data.uniform_(-initial_weight, initial_weight)
        # mean, variance, normal distribution

    def forward(self, state):
        x = self.conv_net(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_variance = self.log_variance_fc(x)
        log_variance = torch.clamp(log_variance, -20, 2)
        return mean, log_variance

    def sample(self, state, epsilon=1e-6):
        mean, log_variance = self.forward(state)
        variance = log_variance.exp()
        gaussian = Normal(mean, variance)        
        z = gaussian.sample()
        
        log_pi = (gaussian.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)).sum(1, keepdim=True)
        return mean, variance, z, log_pi