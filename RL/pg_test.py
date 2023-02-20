import torch
import os
from .agent import Agent


class PGAgent(Agent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu', seed: int = 1) -> None:
        super(PGAgent, self).__init__(state_space_size, action_space_size)
        torch.manual_seed(seed)
        self.model = None
        self.device = device
        self.train_count = 0
        self.log_probs = []
        self.rewards = []

    def create_model(self, model: torch.nn.Module, lr: float = 0.01, y: float = 0.9):
        self.lr = lr
        self.y = y
        self.model = model(self.state_space_size, self.action_space_size)
        self.model.to(self.device)
        self.model.train()
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
        self.step_count += 1
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, log_prob = self.model(state)
        self.log_probs.append(log_prob)
        return action

    def learn(self, reward, episode_over):
        self.rewards.append(reward)
        if episode_over:
            self.update_model()

    def update_model(self):
        self.train_count += 1
        last = 0
        discounted_returns = []
        for item in reversed(self.rewards):
            last = (last * self.y + item)
            discounted_returns.append(last)
        returns = torch.tensor(list(reversed(discounted_returns)))
        # print(returns)
        returns -= returns.mean()
        # print(returns)
        if len(returns) > 1:
            returns /= (returns.std())
        loss = torch.tensor([-log_prob * discounted_return
                             for log_prob, discounted_return
                             in zip(self.log_probs, returns)],
                            requires_grad=True).sum()

        # if self.train_count % 10 == 0:
        print(f"Train: {self.train_count} - loss --------------------------> {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []
