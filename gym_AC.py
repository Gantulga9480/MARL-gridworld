import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from RL.actor_critic import ActorCriticAgent
import gym
import matplotlib.pyplot as plt


class PG(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc_input = nn.Linear(observation_size, 512)
        self.fc_hidden1 = nn.Linear(512, 256)
        self.fc_hidden2 = nn.Linear(256, 128)

        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc_input(x))
        x = F.leaky_relu(self.fc_hidden1(x))
        x = F.leaky_relu(self.fc_hidden2(x))

        probs = F.softmax(self.actor(x), dim=0)
        value = self.critic(x)

        distribution = Categorical(probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action), value


ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME, render_mode=None)
agent = ActorCriticAgent(4, 2, device="cuda:0", seed=42)
agent.create_model(PG, lr=0.00025, y=0.99)
scores = []

while agent.episode_count < 1000:
    reward = []
    done = False
    s, i = env.reset(seed=42)
    while not done:
        a = agent.policy(s)
        s, r, d, t, i = env.step(a)
        done = d or t
        agent.learn(r, done)
        reward.append(r)
    scores.append(sum(reward))
env.close()

plt.plot(scores)
plt.show()

env = gym.make(ENV_NAME, render_mode="human")
for _ in range(10):
    done = False
    s, i = env.reset(seed=42)
    while not done:
        a = agent.policy(s)
        s, r, d, t, i = env.step(a)
        done = d or t
env.close()
