import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from RL.actor_critic import ActorCriticAgent
import gym
import matplotlib.pyplot as plt


class AC(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.actor_input = nn.Linear(observation_size, 512)
        self.actor_hidden1 = nn.Linear(512, 256)
        self.actor_hidden2 = nn.Linear(256, 128)
        self.actor_hidden3 = nn.Linear(128, 64)
        self.critic_input = nn.Linear(observation_size, 512)
        self.critic_hidden1 = nn.Linear(512, 256)
        self.critic_hidden2 = nn.Linear(256, 128)
        self.critic_hidden3 = nn.Linear(128, 64)

        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        ax = F.leaky_relu(self.actor_input(x))
        ax = F.leaky_relu(self.actor_hidden1(ax))
        ax = F.leaky_relu(self.actor_hidden2(ax))
        ax = F.leaky_relu(self.actor_hidden3(ax))
        probs = F.softmax(self.actor(ax), dim=0)
        distribution = Categorical(probs)
        action = distribution.sample()

        cx = F.leaky_relu(self.critic_input(x))
        cx = F.leaky_relu(self.critic_hidden1(cx))
        cx = F.leaky_relu(self.critic_hidden2(cx))
        cx = F.leaky_relu(self.critic_hidden3(cx))
        value = self.critic(cx)
        return action.item(), distribution.log_prob(action), value


torch.manual_seed(3407)
ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME, render_mode=None)
agent = ActorCriticAgent(4, 2, device="cuda:0")
agent.create_model(AC, lr=0.0003, y=0.99)
scores = []

while agent.episode_count < 1000:
    reward = []
    done = False
    s, i = env.reset(seed=3407)
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
