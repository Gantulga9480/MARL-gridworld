from grid import GridEnv
from RL.dqn import DQNAgent
from RL.utils import ReplayBuffer
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, action_size)
        )

    def forward(self, x):
        return self.model(x)


MAX_REPLAY_BUFFER = 1000
BATCH_SIZE = 32
TARGET_NET_UPDATE_FREQ = 10
MAIN_NET_TRAIN_FREQ = 1
CURRENT_TRAIN_ID = '2023-02-11'

environment = GridEnv(env_file="boards/board3.csv")
agent = DQNAgent(environment.observation_size, environment.action_space_size, 0.003, 0.99, e_decay=0.9999, device="cuda:0")
agent.create_model(DQN(environment.observation_size, environment.action_space_size),
                   DQN(environment.observation_size, environment.action_space_size),
                   batchs=BATCH_SIZE,
                   train_freq=MAIN_NET_TRAIN_FREQ,
                   update_freq=TARGET_NET_UPDATE_FREQ)
agent.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, 100))


while environment.running:
    s = environment.reset()
    while not environment.loop_once():
        a = agent.policy(s, greedy=False)
        ns, r, d = environment.step(a)
        agent.learn(s, a, ns, r, d)
        s = ns
        print(agent.e)

print(agent.step_count)
path = '/'.join(['model', CURRENT_TRAIN_ID, 'model'])
agent.save_model(path)
