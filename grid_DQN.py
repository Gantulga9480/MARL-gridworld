from grid import GridEnv, E
from RL.dqn import DQNAgent
from RL.utils import ReplayBufferBase, ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class DQN(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 100),
            nn.ReLU(),
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, action_size)
        )

    def forward(self, x):
        return self.model(x)


class CustomReplayBuffer(ReplayBufferBase):

    def __init__(self, max_size, min_size) -> None:
        super().__init__(max_size, min_size)
        self.buffer_p = deque(maxlen=max_size)
        self.buffer_n = deque(maxlen=max_size)

    @property
    def trainable(self):
        pl = self.buffer_p.__len__()
        nl = self.buffer_n.__len__()
        return pl + nl >= self.min_size

    def push(self, data: list):
        r = data[3]
        if r >= 0:
            self.buffer_p.append(data)
        else:
            self.buffer_n.append(data)

    def extend(self, datas):
        raise NotImplementedError

    def sample(self, sample_size, factor=0.5):
        n_size = round(sample_size * factor)
        p_size = sample_size - n_size
        if self.buffer_n.__len__() >= n_size:
            sn = random.sample(self.buffer_n, n_size)
        else:
            sn = random.sample(self.buffer_n, self.buffer_n.__len__())
        if self.buffer_p.__len__() >= p_size:
            so = random.sample(self.buffer_p, p_size)
        else:
            so = random.sample(self.buffer_p, self.buffer_p.__len__())
        sn.extend(so)
        return sn


MAX_REPLAY_BUFFER = 100
BATCH_SIZE = 100
TARGET_NET_UPDATE_FREQ = 5
MAIN_NET_TRAIN_FREQ = 1
CURRENT_TRAIN_ID = '2023-02-13'

environment = GridEnv(env_file="boards/board3.csv")
agent = DQNAgent(environment.observation_size, environment.action_space_size, 0.003, 0.99, e_decay=0.9995, device="cuda:0")
agent.create_model(DQN(environment.observation_size, environment.action_space_size),
                   DQN(environment.observation_size, environment.action_space_size),
                   batchs=BATCH_SIZE,
                   train_freq=MAIN_NET_TRAIN_FREQ,
                   update_freq=TARGET_NET_UPDATE_FREQ)
agent.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, BATCH_SIZE))
table = np.zeros((8, 8, 4))
environment.model = table

states = []
for i in range(1, 7):
    for j in range(1, 7):
        obj = environment.board[i, j]
        if obj == E:
            up = environment.board[i - 1, j]
            down = environment.board[i + 1, j]
            left = environment.board[i, j - 1]
            right = environment.board[i, j + 1]
            states.append([i, j, up, right, down, left])

state = torch.Tensor(states).to(agent.device)


@torch.no_grad()
def compute_table():
    vals = agent.model(state)
    for i, sta in enumerate(states):
        table[sta[0], sta[1]] = vals[i].to('cpu')


while environment.running:
    s = environment.reset()
    while not environment.loop_once():
        a = agent.policy(s, greedy=False)
        ns, r, d = environment.step(a)
        agent.learn(s, a, ns, r, d)
        compute_table()
        s = ns
    print(agent.e)

print(agent.step_count)
path = '/'.join(['model', CURRENT_TRAIN_ID, 'model'])
agent.save_model(path)
