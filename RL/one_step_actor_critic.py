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
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss(reduction="sum")
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
        if not self.training:
            self.actor.eval()
            with torch.no_grad():
                probs = self.actor(state)
                distribution = Categorical(probs)
                action = distribution.sample()
            return action.item()
        probs = self.actor(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        self.log_prob = distribution.log_prob(action)
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        self.rewards.append(reward)
        self.update_model(state, next_state, reward, episode_over)
        if episode_over:
            self.i = 1
            self.episode_count += 1
            self.step_count = 0
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self, state, next_state, reward, done):
        self.train_count += 1
        self.actor.train()

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        # Bug? It doesn't seem to needed to compute full computational graph when forwarding next_state.
        # But skipping that part breaks learning. Weird!
        # with torch.no_grad():
        next_state_value = (1.0 - done) * self.critic(next_state)
        current_value = self.critic(state)

        critic_loss = self.loss_fn(current_value, reward + self.y * next_state_value)
        critic_loss *= self.i

        # Swapping baseline with Q value for no actor_loss sign
        td_error = current_value.item() - (reward + self.y * next_state_value.item())
        actor_loss = self.log_prob * td_error
        actor_loss *= self.i

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.i *= self.y
