import torch.nn as nn
from torch.distributions import Categorical
from grid import GridEnv
from RL.reinforce import ReinforceAgent
import matplotlib.pyplot as plt


class PG(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        x = self.model(state)
        m = Categorical(x)
        action = m.sample()
        return action.item(), m.log_prob(action)


env = GridEnv(env_file="boards/board3.csv")
agent = ReinforceAgent(env.observation_size, env.action_space_size, device="cuda:0")
agent.create_model(PG, lr=0.0001, y=0.99)

while env.running:
    s = env.reset()
    while not env.loop_once():
        a = agent.policy(s)
        ns, r, d = env.step(a)
        agent.learn(s, a, ns, r, d)
        s = ns

plt.plot(agent.reward_history)
plt.show()

env.running = True

while env.running:
    s = env.reset()
    while not env.loop_once():
        a = agent.policy(s)
        s, r, d = env.step(a)
