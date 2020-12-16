import torch
from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
from collections import deque
from sac import SoftActorCriticAgent
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

env = ObstacleTowerEnv(retro=False, realtime_mode=False)
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
for episode in range(400):
    timesteps = 0
    rewards = 0
    for steps in range(10000):
        timesteps += 1
        actions, actions_env_format = agent.select_actions(state)
        next_state, reward, done, info = env.step(actions_env_format)
        next_state = next_state[0]
        next_state = TF.to_tensor(next_state)
        agent.replay_buffer.add((state, actions, reward, next_state, done))    
        agent.train()
        state = next_state
        rewards += reward
        if(done):
            break
    scores.append(rewards)
    mean_scores_100.append(rewards)
    print('episode {} frames {} rewards {:.2f} mean score {:.2f}'.format(episode, timesteps, rewards, np.mean(mean_scores_100)))
    if(episode % 100 == 0):
        torch.save(agent.conv_net.state_dict(), 'checkpoints/conv_net_checkpoint_{}.pth'.format(version))
        torch.save(agent.critic_v.state_dict(), 'checkpoints/critic_v_checkpoint_{}.pth'.format(version))
        torch.save(agent.critic_q_1.state_dict(), 'checkpoints/critic_q_1_checkpoint_{}.pth'.format(version))
        torch.save(agent.critic_q_2.state_dict(), 'checkpoints/critic_q_2_checkpoint_{}.pth'.format(version))
        torch.save(agent.actor.state_dict(), 'checkpoints/actor_checkpoint_{}.pth'.format(version))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('results/{}_scores.png'.format(version))

# todos:
# run on openai gym Mountaincar or cartpole to test implementation
# 2 hidden layers, 256 units
# build testrunner.py
# up batch size, paper 256
# up buffer 10^6
# timer
# tweak tests reward scale 5, 20
# set up pc3?
# reduce size of convnetwork, more conv layers
# more pooling
# add support to run on gpu(device)
# upgrade implementation to 2019 paper
# store compressed states(run through convnetwork)
# dropout on convnetwork, batchnorm
# stacked states(to capture movement)?
# seed?
# train longer? 1m-10m frames aka 1k-10k episodes?
# reduce training steps?

# v1 canonical (pc1)
# episode 99 frames 600 rewards 0.00 mean score(100ep) 0.25

# v2 canonical, 400 episodes (pc2)
#env crashed
#episode 152 frames 35 rewards 0.00 mean score(100ep) 0.21

# v3 canonical, 300 episodes, lr 3·10^−4 (pc1)
# still not learning, could be suffering from sparse rewards, will validate algorithm on different env

# v4 canonical, 400 episodes, lr 3·10^−4, batch size 256 (pc2)

# v5 canonical, 400 episodes, lr 3·10^−4, reward scaling * 10 (pc1)

# v6 fc layers in convnetwork, lr, reward scaling * 10 (pc1)

