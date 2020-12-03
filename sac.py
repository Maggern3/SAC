import torch
import numpy
# import torch.functional as F
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torchvision.transforms.functional as TF
from neuralnetwork import StateValueNetwork, ActionValueNetwork, PolicyNetwork, ConvNetwork
from buffer import ReplayBuffer

class SoftActorCriticAgent():
    def __init__(self):        
        torch.autograd.set_detect_anomaly(True)
        self.conv_net = ConvNetwork()
        self.critic_v = StateValueNetwork()
        self.critic_v_target = StateValueNetwork()
        self.critic_q_1 = ActionValueNetwork()
        self.critic_q_2 = ActionValueNetwork()
        self.actor = PolicyNetwork()
        self.actor_optim = optim.Adam(self.actor.parameters())
        self.v_optim = optim.Adam(self.critic_v.parameters())
        self.q1_optim = optim.Adam(self.critic_q_1.parameters())
        self.q2_optim = optim.Adam(self.critic_q_2.parameters())        
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 16 #256
        self.replay_buffer = ReplayBuffer(self.batch_size)
        self.update_target(1)

    def select_actions(self, state):    
        self.actor.eval()
        self.conv_net.eval()
        with torch.no_grad():    
            state = self.conv_net(state.unsqueeze(0))
            mean, log_variance = self.actor.forward(state)
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
        self.conv_net.train()
        return actions, numpy.array(actions_env_format)

    def train(self):   
        if(len(self.replay_buffer.replay_buffer) < self.batch_size):  
            return  
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        states = self.conv_net(states).detach()
        next_states = self.conv_net(next_states).detach()

        current_q_1 = self.critic_q_1(states, actions)
        current_q_2 = self.critic_q_2(states, actions)
        current_critic_v = self.critic_v(states)
        mean, variance, z, log_pi = self.actor.sample(states)
        policy_actions = torch.tanh(z)

        # r(st,at) +γEst+1∼p[V ̄ψ(st+1)],
        target_q = rewards + (self.gamma * self.critic_v_target(next_states) * (1-dones)) 
        q1_loss = F.mse_loss(current_q_1, target_q.detach()) 
        q2_loss = F.mse_loss(current_q_2, target_q.detach())
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        q1 = self.critic_q_1(states, policy_actions)
        q2 = self.critic_q_2(states, policy_actions)
        predicted_new_q = torch.min(q1, q2)

        # Eat∼πφ[Qθ(st,at)−logπφ(at|st)]
        target_critic_v = predicted_new_q - log_pi
        critic_loss = F.mse_loss(current_critic_v, target_critic_v.detach())
        self.v_optim.zero_grad()
        critic_loss.backward()
        self.v_optim.step()

        actor_loss = (log_pi - predicted_new_q).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.update_target(self.tau)

    def update_target(self, tau):
        for target_param, param in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)