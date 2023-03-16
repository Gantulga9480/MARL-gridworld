import torch
from torch.distributions import Categorical
from .deep_agent import DeepAgent
import numpy as np


class OneStepActorCriticAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.actor = None
        self.critic = None
        self.log_prob = None
        self.value = None
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss()
        self.i = 1
        del self.model
        del self.optimizer
        del self.lr

    def create_model(self, actor: torch.nn.Module, critic: torch.nn.Module, actor_lr: float, critic_lr: float, y: float):
        self.y = y
        self.actor = actor(self.state_space_size, self.action_space_size)
        self.actor.to(self.device)
        self.actor.train()
        self.critic = critic(self.state_space_size)
        self.critic.to(self.device)
        self.critic.train()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def policy(self, state):
        self.step_count += 1
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not self.train:
            self.actor.eval()
        probs = self.actor(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        if self.train:
            self.value = self.critic(state)
            self.log_prob = distribution.log_prob(action)
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        self.rewards.append(reward)
        if self.train:
            self.update_model(next_state, reward, episode_over)
        if episode_over:
            self.i = 1
            self.episode_count += 1
            self.step_count = 0
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self, next_state, reward, done):
        self.train_count += 1
        self.actor.train()

        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            if not done:
                current_state_target = reward + self.y * self.critic(next_state)
            else:
                current_state_target = reward

        critic_loss = current_state_target - self.value
        actor_loss = self.log_prob * critic_loss.item()

        self.critic.zero_grad()
        critic_loss.backward()
        for p in self.critic.parameters():
            grad = p.grad() * self.i * 0.01

        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        self. i *= self.y
