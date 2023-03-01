from RL import DeepDeterministicPolicyGradientAgent as DDPGAgent
from RL.utils import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)


class Actor(nn.Module):

    def __init__(self, observation_size, action_size, action_max):
        super().__init__()
        self.action_max = action_max
        self.model = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x) * self.action_max


class Critic(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size + action_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))


MAX_REPLAY_BUFFER = 1_000_000
BATCH_SIZE = 64
MAIN_NET_TRAIN_FREQ = 1
ENV_NAME = "CartPole-v1"

env = gym.make(ENV_NAME, render_mode=None)
agent = DDPGAgent(4, 2, device="cuda:0")
agent.create_model(Actor, Critic, lr=0.0003, y=0.99, e_decay=0.999, batchs=BATCH_SIZE, tau=0.01)
agent.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, BATCH_SIZE * 10, 4))

scores = []
while agent.episode_count < 200:
    reward = []
    done = False
    s, info = env.reset(seed=3407)
    while not done:
        a = agent.policy(s)
        ns, r, d, t, i = env.step(a)
        done = d or t
        agent.learn(s, a, ns, r, done, update="soft")
        s = ns
        reward.append(r)
    r_sum = sum(reward)
    scores.append(r_sum)
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
