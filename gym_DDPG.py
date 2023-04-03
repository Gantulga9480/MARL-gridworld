from RL import DeepDeterministicPolicyGradientAgent as DDPGAgent
from RL.utils import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)


class Actor(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


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
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))


ENV_NAME = "InvertedPendulum-v4"
env = gym.make(ENV_NAME, render_mode=None)
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.n, device="cuda:0")
agent.create_model(Actor, Critic, actor_lr=0.0001, critic_lr=0.0001, y=0.99, noise_std=0.1, batch=64, tau=0.01)
agent.create_buffer(ReplayBuffer(1_000_000, 1000, env.observation_space.shape[0], env.action_space.n))

try:
    while agent.episode_count < 1000:
        done = False
        s, info = env.reset(seed=3407)
        while not done:
            a = agent.policy(s) * 3.0
            ns, r, d, t, i = env.step(a)
            done = d or t
            agent.learn(s, a, ns, r, done)
            s = ns
except KeyboardInterrupt:
    pass
env.close()

plt.plot(agent.reward_history)
plt.show()

with open(f"ddpg_rewards_0.0001_{agent.target_update_rate}.txt", "w") as f:
    f.writelines([str(item) + '\n' for item in agent.reward_history])

agent.train = False

env = gym.make(ENV_NAME, render_mode="human")
for _ in range(10):
    done = False
    s, i = env.reset(seed=3407)
    while not done:
        a = agent.policy(s)
        s, r, d, t, i = env.step(a)
        done = d or t
env.close()
