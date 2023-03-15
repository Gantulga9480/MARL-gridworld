import torch
from torch.distributions import Categorical
from .deep_agent import DeepAgent
import numpy as np


class ActorCriticAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.actor = None
        self.critic = None
        self.log_probs = []
        self.values = []
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss()
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
            value = self.critic(state)
            log_prob = distribution.log_prob(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        if self.train:
            self.rewards.append(reward)
            if episode_over:
                self.episode_count += 1
                self.reward_history.append(np.sum(self.rewards))
                self.update_model()
                print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self):
        self.train_count += 1
        self.actor.train()
        G = []
        r_sum = 0
        for r in reversed(self.rewards):
            r_sum = r_sum * self.y + r
            G.append(r_sum)
        G = torch.tensor(list(reversed(G)), dtype=torch.float32).to(self.device)
        # G -= G.mean()
        # if len(G) > 1:
        #     G /= (G.std() + self.eps)

        V = torch.cat(self.values)

        with torch.no_grad():
            A = G - V

        actor_loss = torch.stack([-log_prob * a for log_prob, a in zip(self.log_probs, A)]).sum()
        critic_loss = self.loss_fn(V, G)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
