import torch
import torch.functional as F
import torch.optim as optim
from torch.distributions import Normal
from neuralnetwork import NeuralNetwork, NeuralNetwork2, NeuralNetwork3, ConvClass
from buffer import ReplayBuffer

class SoftActorCriticAgent():
    def __init__(self):        
        self.conv = ConvClass()
        self.critic_v = NeuralNetwork(self.conv)
        self.critic_v_target = NeuralNetwork(self.conv)
        self.critic_q_1 = NeuralNetwork2(self.conv)
        self.critic_q_2 = NeuralNetwork2(self.conv)
        self.actor = NeuralNetwork3(self.conv)
        self.actor_optim = optim.Adam(self.actor.parameters())
        self.v_optim = optim.Adam(self.critic_v.parameters())
        self.q1_optim = optim.Adam(self.critic_v.parameters())
        self.q2_optim = optim.Adam(self.critic_v.parameters())        
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.replay_buffer = ReplayBuffer(self.batch_size)
        self.update_target(1)

    def add(self, sars):
        self.replay_buffer.append(sars)

    def select_actions(self, state):
        #state = torch.from_numpy(state).float().to(self.device)
        state = torch.tensor(state).float()
        mean, log_variance = self.actor.forward(state)
        variance = log_variance.exp()
        gaussian = Normal(mean, variance)        
        z = gaussian.sample()
        actions = torch.tanh(z)
        return actions

    def train(self):   
        if(len(self.replay_buffer) < self.batch_size):  
            return  
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        current_critic_v = self.critic_v(states)
        mean, variance, z, log_pi = self.actor.sample(states)
        q1 = self.critic_q_1(states, actions)
        q2 = self.critic_q_2(states, actions)
        # Eat∼πφ[Qθ(st,at)−logπφ(at|st)]
        target_critic_v = min(q1, q2) - log_pi
        critic_loss = F.mse_loss(current_critic_v, target_critic_v)
        self.v_optim.zero_grad()
        critic_loss.backward()
        self.v_optim.step()

        # r(st,at) +γEst+1∼p[V ̄ψ(st+1)],
        target_q = rewards + (self.gamma * self.critic_v_target(next_states) * (1-dones)) 
        current_q = self.critic_q_1(states, actions)
        q1_loss = F.mse_loss(current_q, target_q) 
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        q2_loss = F.mse_loss(current_q, target_q)
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        policy_actions = torch.tanh(z)
        q1 = self.critic_q_1(states, policy_actions)
        q2 = self.critic_q_2(states, policy_actions)
        actor_loss = (log_pi - min(q1, q2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        update_target(self.tau)

    def update_target(self, tau):
        for target_param, param in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)