import torch
import numpy
from torch._C import device
# import torch.functional as F
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torchvision.transforms.functional as TF
from neuralnetwork import StateValueNetwork, ActionValueNetwork, PolicyNetwork, ConvNetwork
from buffer import ReplayBuffer

class SoftActorCriticAgent():
    def __init__(self):        
        #torch.autograd.set_detect_anomaly(True)
        gpu = torch.cuda.is_available()
        if(gpu):
            print('GPU/CUDA works! Happy fast training :)')
            torch.cuda.current_device()
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
        else:
            print('training on cpu...')
            self.device = torch.device("cpu")
        self.critic_q_1 = ActionValueNetwork().to(self.device)
        self.critic_q_1_target = ActionValueNetwork().to(self.device)
        self.critic_q_2 = ActionValueNetwork().to(self.device)
        self.critic_q_2_target = ActionValueNetwork().to(self.device)
        self.actor = PolicyNetwork().to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3*10e-4) #0.003
        self.q1_optim = optim.Adam(self.critic_q_1.parameters(), lr=0.003)
        self.q2_optim = optim.Adam(self.critic_q_2.parameters(), lr=0.003)        
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.reward_scale = 10
        self.replay_buffer = ReplayBuffer(self.batch_size, self.device)
        self.update_target(1)

    def select_actions(self, state):    
        self.actor.eval()
        with torch.no_grad():    
            mean, log_variance = self.actor.forward(state.unsqueeze(0).to(self.device))
            variance = log_variance.exp()
            gaussian = Normal(mean, variance)        
            z = gaussian.sample()
            actions = torch.tanh(z)
            actions = actions.cpu().detach().squeeze(0)
            dim1 = actions[0:3]
            dim1_p = F.softmax(dim1, 0)
            action1 = torch.argmax(dim1_p)
            dim2 = actions[3:6]
            dim2_p = F.softmax(dim2, 0)
            action2 = torch.argmax(dim2_p)
            dim3 = actions[6:8]
            dim3_p = F.softmax(dim3, 0)
            action3 = torch.argmax(dim3_p)
            dim4 = actions[8:11]
            dim4_p = F.softmax(dim4, 0)
            action4 = torch.argmax(dim4_p)
            actions_env_format = [action1.item(), action2.item(), action3.item(), action4.item()]
        self.actor.train()
        return actions, numpy.array(actions_env_format)

    def train(self):   
        if(len(self.replay_buffer.replay_buffer) < self.batch_size):  
            return  
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        current_q_1 = self.critic_q_1(states, actions)
        current_q_2 = self.critic_q_2(states, actions)
        

        next_mean, next_variance, next_z, next_log_pi = self.actor.sample(next_states)
        next_policy_actions = torch.tanh(next_z)
        next_q = min(self.critic_q_1_target(next_states, next_policy_actions), self.critic_q_2_target(next_states, next_policy_actions))
        next_q = next_q - self.alpha * next_log_pi
        target_q = rewards * self.reward_scale + (self.gamma * next_q * (1-dones)) 
        q1_loss = F.mse_loss(current_q_1, target_q.detach()) 
        q2_loss = F.mse_loss(current_q_2, target_q.detach())
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        mean, variance, z, log_prob = self.actor.sample(states)
        policy_actions = torch.tanh(z)
        q1 = self.critic_q_1(states, policy_actions)
        q2 = self.critic_q_2(states, policy_actions)
        predicted_new_q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_prob - predicted_new_q).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.update_target(self.tau)

    def update_target(self, tau):
        for target_param, param in zip(self.critic_q_1_target.parameters(), self.critic_q_1.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
        for target_param, param in zip(self.critic_q_2_target.parameters(), self.critic_q_2.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)