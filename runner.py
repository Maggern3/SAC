from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
from collections import deque
from sac import SoftActorCriticAgent
import torchvision.transforms.functional as TF

env = ObstacleTowerEnv(retro=False, realtime_mode=True)
print(env.action_space)
print(env.observation_space)

agent = SoftActorCriticAgent()

state = env.reset()
#print(state.shape)
state = state[0]
state = TF.to_tensor(state)
print(state.size)
#for episode in range(10):
for steps in range(1000):
    actions = agent.select_actions(state)
    next_state, reward, done, info = env.step(actions)
    next_state = next_state[0]
    next_state = TF.to_tensor(next_state)
    agent.replay_buffer.add((state, actions, reward, next_state, done))    
    agent.train()
    state = next_state
    if(done):
        break