import torch.nn as nn
from torch.distributions import Categorical
from grid import GridEnv
from RL.reinforce import ReinforceAgent
import matplotlib.pyplot as plt


class PG(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, action_size),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        x = self.model(state)
        m = Categorical(x)
        action = m.sample()
        return action.item(), m.log_prob(action)


env = GridEnv(env_file="boards/board4.csv")
agent = ReinforceAgent(env.observation_size, env.action_space_size, device="cuda:0")
agent.create_model(PG, lr=0.00025, y=0.99)

scores = []

while env.running:
    s = env.reset()
    rewards = []
    while not env.loop_once():
        a = agent.policy(s)
        s, r, d = env.step(a)
        agent.learn(r, d)
        rewards.append(r)
    scores.append(sum(rewards))
    if agent.train_count >= 1000:
        env.running = False

plt.plot(scores)
plt.show()
