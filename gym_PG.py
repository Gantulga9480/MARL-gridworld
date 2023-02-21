import torch
import torch.nn as nn
from torch.distributions import Categorical
from RL.pg_test import PGAgent
import gym
import matplotlib.pyplot as plt


class PG(torch.nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(observation_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, action_size),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.model(x)
        m = Categorical(x)
        action = m.sample()
        return action.item(), m.log_prob(action)


env = gym.make("CartPole-v1")
agent = PGAgent(4, 2, device="cuda:0", seed=42)
agent.create_model(PG, lr=0.01, y=1)
scores = []
reward = []
s, i = env.reset(seed=42)
while agent.episode_count < 3000:
    a = agent.policy(s)
    s, r, d, f, i = env.step(a)
    reward.append(r)
    agent.learn(r, d or f)
    if d or f:
        scores.append(sum(reward))
        reward = []
        s, info = env.reset()
env.close()

plt.plot(scores)
plt.show()

env = gym.make("CartPole-v1", render_mode="human")
s, i = env.reset(seed=42)
for _ in range(1000):
    a = agent.policy(s)
    s, r, d, f, i = env.step(a)
    if d or f:
        s, info = env.reset()
env.close()
