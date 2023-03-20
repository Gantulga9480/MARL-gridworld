import torch
import torch.nn as nn
from RL.one_step_actor import OneStepActor
import gym
import matplotlib.pyplot as plt
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)


class PG(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)


ENV_NAME = "CartPole-v1"
TRAIN_ID = "OSPG Test"
env = gym.make(ENV_NAME, render_mode=None)
agent = OneStepActor(env.observation_space.shape[0], env.action_space.n, device="cuda:0")
agent.create_model(PG, lr=0.0001, y=0.99)

try:
    while agent.episode_count < 1000:
        done = False
        s, i = env.reset(seed=3407)
        while not done:
            a = agent.policy(s)
            ns, r, d, t, i = env.step(a)
            done = d or t
            agent.learn(s, a, ns, r, done)
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
