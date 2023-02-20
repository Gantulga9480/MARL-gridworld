from RL.dqn import DQNAgent
from RL.utils import ReplayBuffer
import torch.nn as nn
import gym


class DQN(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


MAX_REPLAY_BUFFER = 10000
BATCH_SIZE = 64
TARGET_NET_UPDATE_FREQ = 5
MAIN_NET_TRAIN_FREQ = 1

env = gym.make("CartPole-v1", render_mode="human")
agent = DQNAgent(4, 2, device="cuda:0", seed=42)
agent.create_model(DQN, lr=0.00025, y=0.99, e_decay=0.9975, batchs=BATCH_SIZE, main_train_freq=MAIN_NET_TRAIN_FREQ, target_update_freq=TARGET_NET_UPDATE_FREQ)
agent.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, 1000))


s, info = env.reset(seed=42)
done = False
while True:
    a = agent.policy(s)
    ns, r, d, f, i = env.step(a)
    agent.learn(s, a, ns, r, d)
    s = ns
    if d:
        s, info = env.reset()
