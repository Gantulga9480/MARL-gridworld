import torch
import numpy as np
import os
from .agent import Agent
from .utils import ReplayBufferBase


class PGAgent(Agent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu', seed: int = 1) -> None:
        super(PGAgent, self).__init__(state_space_size, action_space_size)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = None
        self.buffer = None
        self.batchs = 0
        self.device = device
        self.main_train_freq = 0
        self.train_count = 0
        self.log_probs = []

    def create_buffer(self, buffer: ReplayBufferBase):
        if buffer.min_size == 0:
            buffer.min_size = self.batchs
        self.buffer = buffer

    def create_model(self, model: torch.nn.Module, lr: float = 0.0003, y: float = 0.9, batchs: int = 64):
        self.lr = lr
        self.y = y
        self.model = model(self.state_space_size, self.action_space_size)
        self.model.to(self.device)
        self.model.train()
        self.batchs = batchs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def save_model(self, path: str) -> None:
        if self.model and path:
            try:
                torch.save(self.model.state_dict(), path)
            except Exception:
                os.makedirs("/".join(path.split("/")[:-1]))
                torch.save(self.model.state_dict(), path)

    def load_model(self, path) -> None:
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)
            self.model.train()
        except Exception:
            print(f'{path} file not found!')
            exit()

    def policy(self, state):
        """greedy - False (default) for training, True for inference"""
        self.step_count += 1
        state = torch.Tensor(state).to(self.device)
        action, log_prob = self.model(state)
        self.log_probs.append(log_prob)
        return action

    def learn(self, reward, episode_over):
        batch = len(np.array(reward).shape) > 1
        if not batch:
            self.buffer.push([reward, episode_over])
        else:
            self.buffer.extend([reward, episode_over])
        if episode_over:
            self.update_model()

    def update_model(self):
        self.train_count += 1
        last = 0
        discounted_returns = []
        for item in reversed(self.buffer.buffer):
            last = (last * self.y + item[0])
            discounted_returns.append(last)
        self.buffer.clear()
        returns = torch.tensor(list(reversed(discounted_returns)))
        returns -= returns.mean()
        returns /= (returns.std() + np.finfo(np.float32).eps.item())

        loss = []
        for log_prob, discounted_return in zip(self.log_probs, returns):
            loss.append(-log_prob * discounted_return)
        loss = torch.cat(loss).sum()
        self.log_probs = []

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.train_count % 10 == 0:
            print(f"Train: {self.train_count} - loss ---> ", loss.item())
