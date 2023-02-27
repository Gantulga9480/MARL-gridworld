import os
import torch
import numpy as np
from .deep_agent import DeepAgent
from .utils import ReplayBufferBase


class DDPGAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.actor = None
        self.target_actor = None
        self.critic = None
        self.target_critic = None
        self.buffer = None
        self.batchs = 0
        self.target_update_rate = 0
        self.train_count = 0
        self.loss_fn = torch.nn.HuberLoss()

    def create_buffer(self, buffer: ReplayBufferBase):
        if buffer.min_size == 0:
            buffer.min_size = self.batchs
        self.buffer = buffer

    def create_model(self, actor: torch.nn.Module, critic: torch.nn.Module, lr: float, y: float, batch: int = 64, tau: float = 0.001):
        self.lr = lr
        self.y = y
        self.batchs = batch
        self.target_update_rate = tau
        self.actor = actor(self.state_space_size, self.action_space_size)
        self.target_actor = actor(self.state_space_size, self.action_space_size)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor.to(self.device)
        self.actor.train()
        self.target_actor.to(self.device)
        self.target_actor.eval()
        self.critic = critic(self.state_space_size)
        self.target_critic = critic(self.state_space_size)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic.to(self.device)
        self.critic.train()
        self.target_critic.to(self.device)
        self.target_critic.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    @torch.no_grad()
    def policy(self, state, greedy=False):
        """greedy - False (default) for training, True for inference"""
        self.step_count += 1
        self.actor.eval()
        state = torch.Tensor(state).to(self.device)
        if not greedy and np.random.random() < self.e:
            return np.random.choice(list(range(self.action_space_size)))
        else:
            return torch.argmax(self.actor(state)).item()

    def learn(self, state, action, next_state, reward, episode_over):
        self.buffer.push(state, action, next_state, reward, episode_over)
        if self.buffer.trainable and self.train:
            self.update_model()
            self.update_target()
            self.decay_epsilon()

    def update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.target_update_rate * local_param.data + (1.0 - self.target_update_rate) * target_param.data)

        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * local_param.data + (1.0 - self.target_update_rate) * target_param.data)

    def update_model(self):
        s, a, ns, r, d = self.buffer.sample(self.batchs)
        self.train_count += 1
        self.critic.eval()
        states = torch.tensor(s, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(ns, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            current_qs = self.critic(states)
            future_qs = self.target_critic(next_states)
            for i in range(len(s)):
                if not d[i]:
                    new_q = r[i] + self.y * torch.max(future_qs[i])
                else:
                    new_q = r[i]
                current_qs[i][a[i]] = new_q.item()

        self.critic.train()
        preds = self.critic(states)
        loss = self.loss_fn(preds, current_qs).to(self.device)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        

        if self.train_count % 100 == 0:
            print(f"Train: {self.train_count} - loss ---> ", loss.item())
