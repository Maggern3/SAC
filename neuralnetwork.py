import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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

class ActionValueNetwork(nn.Module):
    def __init__(self):    
        super().__init__()#NeuralNetwork, self
        self.conv_net = ConvNetwork()#.to(self.device)
        self.fc1 = nn.Linear(convoutputsize, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 11)
        # uniform init layer 3
        self.fc3.weight.data.uniform_(-initial_weight, initial_weight)
        self.fc3.bias.data.uniform_(-initial_weight, initial_weight)

    def forward(self, state):
        x = self.conv_net(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self):    
        super().__init__()#NeuralNetwork, self
        self.conv_net = ConvNetwork()#.to(self.device)
        self.fc1 = nn.Linear(convoutputsize, 256)
        self.fc2 = nn.Linear(256, 11)

        self.fc2.weight.data.uniform_(-initial_weight, initial_weight)
        self.fc2.bias.data.uniform_(-initial_weight, initial_weight)
        # mean, variance, normal distribution

    def forward(self, state):
        x = self.conv_net(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #log_prob = F.log_softmax(x)
        #dim1 = actions[0:3]
        dim1_p = F.log_softmax(x[:,0:3], dim=1)
        action1 = Categorical(dim1_p).sample()
        dim2_p = F.log_softmax(x[:,3:6], dim=1)
        action2 = Categorical(dim2_p).sample()
        dim3_p = F.log_softmax(x[:,6:8], dim=1)
        action3 = Categorical(dim3_p).sample()
        dim4_p = F.log_softmax(x[:,8:11], dim=1)
        action4 = Categorical(dim4_p).sample()
        actions_env_format = [action1.item(), action2.item(), action3.item(), action4.item()]
        log_prob = torch.cat((dim1_p, dim2_p, dim3_p, dim4_p), dim=1)
        return log_prob, actions_env_format