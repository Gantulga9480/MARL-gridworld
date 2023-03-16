import torch
import torch.nn as nn
from RL.one_step_actor_critic import OneStepActorCriticAgent
import gym
import matplotlib.pyplot as plt
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)


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
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):

    def __init__(self, observation_size) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


ENV_NAME = "CartPole-v1"
TRAIN_ID = "OSAC test"
env = gym.make(ENV_NAME, render_mode=None)
agent = OneStepActorCriticAgent(4, 2, device="cuda:0")
agent.create_model(Actor, Critic, actor_lr=0.001, critic_lr=0.01, y=0.99)

try:
    while agent.episode_count < 1000:
        done = False
        s, i = env.reset(seed=3407)
        while not done:
            a = agent.policy(s)
            ns, r, d, t, i = env.step(a)
            done = d or t
            agent.learn(s, r, ns, r, done)
            s = ns
except KeyboardInterrupt:
    pass
env.close()

plt.xlabel(f"{ENV_NAME} - {TRAIN_ID}")
plt.plot(agent.reward_history)
plt.show()

# agent.train = False
# env = gym.make(ENV_NAME, render_mode="human")
# for _ in range(10):
#     done = False
#     s, i = env.reset(seed=3407)
#     while not done:
#         a = agent.policy(s)
#         s, r, d, t, i = env.step(a)
#         done = d or t
# env.close()
