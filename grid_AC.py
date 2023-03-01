import torch.nn as nn
from torch.distributions import Categorical
from grid import GridEnv
from RL import ActorCriticAgent
import matplotlib.pyplot as plt


class AC(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=0)
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        probs = self.actor(x)
        distribution = Categorical(probs)
        action = distribution.sample()

        value = self.critic(x)
        return action.item(), distribution.log_prob(action), value


env = GridEnv(env_file="boards/board3.csv")
agent = ActorCriticAgent(env.observation_size, env.action_space_size, device="cuda:0")
agent.create_model(AC, lr=0.00025, y=0.99)

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
    if agent.train_count >= 5000:
        env.running = False

plt.plot(scores)
plt.show()

env.running = True
scores = []

while env.running:
    s = env.reset()
    rewards = []
    while not env.loop_once():
        a = agent.policy(s)
        s, r, d = env.step(a)
        rewards.append(r)
    scores.append(sum(rewards))

plt.plot(scores)
plt.show()
