import torch
import torch.nn as nn
from torch.distributions import Categorical
from RL.pg_test import PGAgent
import gym


class PG(torch.nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.model(x)
        m = Categorical(x)
        action = m.sample()
        return action.item(), m.log_prob(action)


env = gym.make("CartPole-v1", render_mode="human")
agent = PGAgent(4, 2, device="cuda:0", seed=42)
agent.create_model(PG, lr=0.00025, y=0.99)

s, i = env.reset(seed=42)
while True:
    a = agent.policy(s)
    s, r, d, f, i = env.step(a)
    agent.learn(r, d)
    if d:
        s, info = env.reset()
