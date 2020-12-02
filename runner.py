from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
from collections import deque
from sac import SoftActorCriticAgent

env = ObstacleTowerEnv(retro=False, realtime_mode=True)
print(env.action_space)
print(env.observation_space)

agent = SoftActorCriticAgent()

state = env.reset()[0]

print(state.shape)
#for episode in range(10):
for steps in range(1000):
    actions = agent.select_actions(state)
    next_state, reward, done, info = env.step(actions)
    agent.add((state, actions, reward, next_state, done))    
    agent.train()
    state = next_state