from grid import GridEnv
from RL.dqn import DQNAgent
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class DQN(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


env = GridEnv(env_file="boards/board2.csv")
agent = DQNAgent(env.observation_size, env.action_space_size)
agent.create_model(DQN)
agent.load_model('model_dqn.pt')
scores = []
eps_count = 0

while env.running:
    eps_count += 1
    s = env.reset()
    rewards = []
    while not env.loop_once():
        a = agent.policy(s, greedy=True)
        ns, r, d = env.step(a)
        s = ns
        rewards.append(r)
    if eps_count > 1000:
        env.running = False

    scores.append(np.sum(rewards))

plt.plot(scores)
plt.show()
