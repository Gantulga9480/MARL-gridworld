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
        self.loss_fn = torch.nn.HuberLoss(reduction='sum')
        self.reward_norm_factor = 1.0
        del self.model
        del self.optimizer
        del self.lr

    def create_model(self, actor: torch.nn.Module, critic: torch.nn.Module, actor_lr: float, critic_lr: float, y: float, reward_norm_factor: float = 1.0):
        self.y = y
        self.reward_norm_factor = reward_norm_factor
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
        self.rewards.append(reward)
        if episode_over:
            self.episode_count += 1
            self.step_count = 0
            self.reward_history.append(np.sum(self.rewards))
            if self.update_model():
                print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")
            self.rewards.clear()

    def update_model(self):
        if len(self.rewards) <= 1:
            return False
        self.train_count += 1
        self.actor.train()
        G = torch.tensor(self.rewards).float()
        G /= self.reward_norm_factor
        r_sum = 0
        for i in reversed(range(G.shape[0])):
            r_sum = r_sum * self.y + G[i]
            G[i] = r_sum
        G.to(self.device)
        G -= G.mean()
        G /= (G.std() + self.eps)

        V = torch.cat(self.values)

        with torch.no_grad():
            A = V - G  # swapping position for negative sign

        actor_loss = torch.stack([log_prob * a for log_prob, a in zip(self.log_probs, A)]).sum()
        critic_loss = self.loss_fn(V, G)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.log_probs.clear()
        self.values.clear()
        return True
