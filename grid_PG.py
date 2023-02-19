import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from grid import GridEnv
from RL.pg_test import PGAgent
from RL.utils import ReplayBuffer


class PG(torch.nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(observation_size, 100)
        self.fc2 = torch.nn.Linear(100, action_size)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        try:
            m = Categorical(x)
        except ValueError:
            for param in self.fc1.parameters():
                print(param)
            raise ValueError()
        action = m.sample()
        return action.item(), m.log_prob(action)


env = GridEnv(env_file="boards/board1.csv")
agent = PGAgent(env.observation_size, env.action_space_size, device="cuda:0")
agent.create_model(PG)
agent.create_buffer(ReplayBuffer(1000, 1000))

rewards = []

while env.running:
    s = env.reset()
    while not env.loop_once():
        a = agent.policy(s)
        ns, r, d = env.step(a)
        agent.learn(r, d)
        s = ns
        rewards.append(r)

with open('reward.txt', 'w') as f:
    f.write('\n'.join([str(item) for item in rewards]))
