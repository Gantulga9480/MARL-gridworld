import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from grid import GridEnv
from RL.pg_test import PGAgent


class PG(torch.nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(observation_size, 16)
        self.fc2 = torch.nn.Linear(16, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=0)
        m = Categorical(x)
        action = m.sample()
        return action.item(), m.log_prob(action)


env = GridEnv(env_file="boards/board3.csv")
agent = PGAgent(env.observation_size, env.action_space_size, device="cuda:0")
agent.create_model(PG, lr=0.1, y=0.9)

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
