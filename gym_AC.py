import torch
import torch.nn as nn
from torch.distributions import Categorical
from RL.actor_critic import ActorCriticAgent
import gym
import matplotlib.pyplot as plt
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)


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
            nn.Linear(512, 1)
        )

    def forward(self, x):
        probs = self.actor(x)
        distribution = Categorical(probs)
        action = distribution.sample()

        value = self.critic(x)
        return action.item(), distribution.log_prob(action), value


ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME, render_mode=None)
agent = ActorCriticAgent(4, 2, device="cuda:0")
agent.create_model(AC, lr=0.0001, y=0.99)
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
    s, i = env.reset(seed=3407)
    while not done:
        a = agent.policy(s)
        s, r, d, t, i = env.step(a)
        done = d or t
env.close()
