
from obstacle_tower_env import ObstacleTowerEnv
#from unityagents import UnityEnvironment
from matplotlib import pyplot as plt
from collections import deque
from neuralnetwork import NeuralNetwork, NeuralNetwork2, ConvClass

env = ObstacleTowerEnv(retro=False, realtime_mode=True)
#env = UnityEnvironment(file_name='ObstacleTower/ObstacleTower.exe')#, retro=False, realtime_mode=False)
print(env.action_space)
print(env.observation_space)
state = env.reset()
replayBuffer = deque(len=10000)
# three to four neural networks, fc + 3 conv layers
actor = NeuralNetwork()
conv = ConvClass()
critic_v = NeuralNetwork(conv)
critic_v_target = NeuralNetwork(conv)
critic_q_1 = NeuralNetwork2(conv)
critic_q_2 = NeuralNetwork2(conv)
#for episode in range(10):
for steps in range(1000):
    actions = actor(state)
    next_state, reward, done, info = env.step(actions)
    replayBuffer.add((state, actions, reward, next_state))
    # train loop
    train()
    state = next_state
    
# loop running through env and training
# criterion(loss, MSELoss?), optimizers(adam)
import torch
import torch.functional as F
import torch.optim as optim

actor_optim = optim.Adam(actor.parameters())
v_optim = optim.Adam(critic_v.parameters())
q1_optim = optim.Adam(critic_v.parameters())
q2_optim = optim.Adam(critic_v.parameters())

def train(self):
    gamma = 0.99
    state, actions, reward, next_state, dones = replayBuffer[0]

    current_critic_v = critic_v(state)
    mean, variance, z, log_pi = actor.sample(state)
    q1 = critic_q_1(state, actions)
    q2 = critic_q_2(state, actions)
    # Eat∼πφ[Qθ(st,at)−logπφ(at|st)]
    target_critic_v = min(q1, q2) - log_pi
    critic_loss = F.mse_loss(current_critic_v, target_critic_v)
    v_optim.zero_grad()
    critic_loss.backward()
    v_optim.step()

    # r(st,at) +γEst+1∼p[V ̄ψ(st+1)],
    target_q = reward + (gamma * critic_v_target(next_state) * (1-dones)) 
    current_q = critic_q_1(state, actions)
    q1_loss = F.mse_loss(current_q, target_q) 
    q1_optim.zero_grad()
    q1_loss.backward()
    q1_optim.step()
    q2_loss = F.mse_loss(current_q, target_q)
    q2_optim.zero_grad()
    q2_loss.backward()
    q2_optim.step()

    policy_actions = torch.tanh(z)
    q1 = critic_q_1(state, policy_actions)
    q2 = critic_q_2(state, policy_actions)
    actor_loss = (log_pi - min(q1, q2)).mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()
#actor_optim.zero_grad()
#update target network