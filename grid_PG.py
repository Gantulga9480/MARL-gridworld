import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from grid import GridEnv
from RL.pg_test import PGAgent


class PG(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
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
        print(x)
        m = Categorical(x)
        action = m.sample()
        return action.item(), m.log_prob(action)


env = GridEnv(env_file="boards/board4.csv")
agent = PGAgent(env.observation_size, env.action_space_size, device="cuda:0")
agent.create_model(PG, lr=0.001, y=0.99)

rewards = []

while env.running:
    s = env.reset()
    ep_r = []
    while not env.loop_once():
        a = agent.policy(s)
        s, r, d = env.step(a)
        agent.learn(r, d)
        ep_r.append(r)
    rewards.append(sum(ep_r))

with open('reward.txt', 'w') as f:
    f.write('\n'.join([str(item) for item in rewards]))
