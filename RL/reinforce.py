import torch
from torch.distributions import Categorical
from .deep_agent import DeepAgent
import numpy as np


class ReinforceAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.log_probs = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()

    def policy(self, state):
        self.step_count += 1
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not self.train:
            self.model.eval()
        probs = self.model(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        if self.train:
            log_prob = distribution.log_prob(action)
            self.log_probs.append(log_prob)
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        self.rewards.append(reward)
        if episode_over:
            self.episode_count += 1
            self.reward_history.append(np.sum(self.rewards))
            if self.train:
                self.update_model()
            else:
                self.rewards = []
            print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self):
        self.train_count += 1
        self.model.train()
        G = []
        r_sum = 0
        for r in reversed(self.rewards):
            r_sum = r_sum * self.y + r
            G.append(r_sum)
        G = torch.tensor(list(reversed(G)), dtype=torch.float32)
        G -= G.mean()
        if len(G) > 1:
            G /= (G.std() + self.eps)

        loss = torch.stack([-log_prob * a for log_prob, a in zip(self.log_probs, G)]).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []
