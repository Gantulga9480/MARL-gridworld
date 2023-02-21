from grid import GridEnv, E
from RL.dqn import DQNAgent
from RL.utils import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class DQN(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


MAX_REPLAY_BUFFER = 1000
BATCH_SIZE = 256
TARGET_NET_UPDATE_FREQ = 5
MAIN_NET_TRAIN_FREQ = 1

env = GridEnv(env_file="boards/board2.csv")
agent = DQNAgent(env.observation_size, env.action_space_size, device="cuda:0")
agent.create_model(DQN, lr=0.00025, y=0.99, e_decay=0.9975, batchs=BATCH_SIZE, main_train_freq=MAIN_NET_TRAIN_FREQ, target_update_freq=TARGET_NET_UPDATE_FREQ)
agent.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, BATCH_SIZE))
table = np.zeros((10, 13, 4))
env.model = table
scores = []

states = []
for i in range(1, 9):
    for j in range(1, 12):
        obj = env.board[i, j]
        if obj == E:
            up = env.board[i - 1, j]
            down = env.board[i + 1, j]
            left = env.board[i, j - 1]
            right = env.board[i, j + 1]
            states.append([i, j, up, right, down, left])

state = torch.Tensor(states).to(agent.device)


@torch.no_grad()
def compute_table():
    vals = agent.model(state)
    for i, sta in enumerate(states):
        table[sta[0], sta[1]] = vals[i].to('cpu')


while env.running:
    s = env.reset()
    rewards = []
    while not env.loop_once():
        a = agent.policy(s, greedy=True)
        ns, r, d = env.step(a)
        agent.learn(s, a, ns, r, d)
        compute_table()
        s = ns

        rewards.append(r)
    scores.append(np.sum(rewards))
    if agent.train_count >= 1000:
        env.running = False

agent.save_model("model_dqn.pt")

plt.plot(scores)
plt.show()
