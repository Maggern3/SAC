import torch
from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
from collections import deque
from sac import SoftActorCriticAgent
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

env = ObstacleTowerEnv(retro=False, realtime_mode=True)
print(env.action_space)
print(env.observation_space)

agent = SoftActorCriticAgent()

state = env.reset()
#print(state.shape)
state = state[0]
state = TF.to_tensor(state)
print(state.size)
scores = []
mean_scores_100 = deque(maxlen=100)
version = 'v3'
agent.conv_net.load_state_dict(torch.load('checkpoints/conv_net_checkpoint_{}.pth'.format(version)))
agent.critic_v.load_state_dict(torch.load('checkpoints/critic_v_checkpoint_{}.pth'.format(version)))
agent.critic_q_1.load_state_dict(torch.load('checkpoints/critic_q_1_checkpoint_{}.pth'.format(version)))
agent.critic_q_2.load_state_dict(torch.load('checkpoints/critic_q_2_checkpoint_{}.pth'.format(version)))
agent.actor.load_state_dict(torch.load('checkpoints/actor_checkpoint_{}.pth'.format(version)))
for episode in range(10):
    timesteps = 0
    rewards = 0
    for steps in range(10000):
        timesteps += 1
        actions, actions_env_format = agent.select_actions(state)
        next_state, reward, done, info = env.step(actions_env_format)
        next_state = next_state[0]
        next_state = TF.to_tensor(next_state)
        state = next_state
        rewards += reward
        if(done):
            break
    scores.append(rewards)
    mean_scores_100.append(rewards)
    print('episode {} frames {} rewards {:.2f} mean score {:.2f}'.format(episode, timesteps, rewards, np.mean(mean_scores_100)))
    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('results/{}_scores.png'.format(version))