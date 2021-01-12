import torch
import numpy
from torch._C import device
# import torch.functional as F
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torchvision.transforms.functional as TF
from neuralnetwork import ActionValueNetwork, PolicyNetwork, ConvNetwork
from buffer import ReplayBuffer

class SoftActorCriticAgent():
    def __init__(self, action_space):        
        torch.autograd.set_detect_anomaly(True)
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
        self.reward_scale = 1
        self.replay_buffer = ReplayBuffer(self.batch_size, self.device)
        self.expected_entropy = -torch.prod(torch.tensor(action_space.shape).to(self.device)).item() # unsure if this is right for multidiscrete env
        print('target entropy', self.expected_entropy)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp() #0.2#, requires_grad=True, device=self.device)#0.2
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.003)

        self.update_target(1)

    def select_actions(self, state):    
        self.actor.eval()
        with torch.no_grad():    
            log_prob, actions, actions_batch_format = self.actor(state.unsqueeze(0).to(self.device))
            actions_env_format = [actions[0].item(), actions[1].item(), actions[2].item(), actions[3].item()]
            #actions = actions.cpu().detach().squeeze(0)            
        self.actor.train()
        return actions_env_format, numpy.array(actions_env_format)

    def train(self):   
        if(len(self.replay_buffer.replay_buffer) < self.batch_size):  
            return  
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # get the values of the chosen actions in the states
        current_q_1 = self.critic_q_1(states)
        q1_w_actions = self.critic_q_1.getActionValuesForActions(current_q_1, actions)
        current_q_2 = self.critic_q_2(states)
        q2_w_actions = self.critic_q_2.getActionValuesForActions(current_q_2, actions)        

        next_log_prob_selected_actions, next_actions, next_actions_batch_format = self.actor(next_states)   
        # ddpg and original sac 19 uses next_actions to get the q_values 
        q1 = self.critic_q_1_target(next_states)    
        q2 = self.critic_q_2_target(next_states)
        next_q1_4s = self.critic_q_1_target.getActionValuesForActions(q1, next_actions_batch_format)
        next_q2_4s = self.critic_q_2_target.getActionValuesForActions(q2, next_actions_batch_format)
        next_q = torch.min(next_q1_4s, next_q2_4s)
        #next_q = torch.min(q1, q2)
        next_q = next_q - self.alpha * next_log_prob_selected_actions
        #next_q = torch.sum(next_q, dim=1).unsqueeze(-1)
        target_q = rewards + (self.gamma * next_q * (1-dones)) 

        q1_loss = F.mse_loss(q1_w_actions, target_q.detach()) 
        q2_loss = F.mse_loss(q2_w_actions, target_q.detach())
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        next_log_prob_selected_actions, actions_env_format, actions_batch_format = self.actor(states)
        
        current_q_1 = self.critic_q_1(states)
        q1_w_actions = self.critic_q_1.getActionValuesForActions(current_q_1, actions)
        current_q_2 = self.critic_q_2(states)
        q2_w_actions = self.critic_q_2.getActionValuesForActions(current_q_2, actions)
        predicted_new_q = torch.min(q1_w_actions, q2_w_actions)        
        actor_loss = (self.alpha * next_log_prob_selected_actions - predicted_new_q).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # my impl based on formula 18 from paper, crashes
        #alpha_loss = (-self.alpha * (log_prob - self.expected_entropy).detach()).mean()
        # rail-berkeley/softlearning, crashes
        #alpha_loss2 = -1.0 * (self.alpha * (log_prob + self.expected_entropy).detach()).mean()
        # cyoon1729/Policy-Gradient-Methods, alpha is less than 0.0 in 80 episodes
        #alpha_loss3 = (self.log_alpha * (-log_prob - self.expected_entropy).detach()).mean()    
        # vitchyr/rlkit, alpha is less than 0.0 in 52 episodes
        # p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
        alpha_loss4 = -(self.log_alpha * (next_log_prob_selected_actions + self.expected_entropy).detach()).mean()
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer.zero_grad()
        alpha_loss4.backward()
        self.alpha_optimizer.step() 
        
        self.update_target(self.tau)

    def update_target(self, tau):
        for target_param, param in zip(self.critic_q_1_target.parameters(), self.critic_q_1.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
        for target_param, param in zip(self.critic_q_2_target.parameters(), self.critic_q_2.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)