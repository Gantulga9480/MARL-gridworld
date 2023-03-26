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
        self.entrops = []
        self.values = []
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss(reduction='sum')
        self.reward_norm_factor = 1.0
        self.entropy_lr = 0.1
        del self.model
        del self.optimizer
        del self.lr

    def create_model(self,
                     actor: torch.nn.Module,
                     critic: torch.nn.Module,
                     actor_lr: float,
                     critic_lr: float,
                     y: float,
                     entropy_lr: float,
                     reward_norm_factor: float = 1.0):
        self.y = y
        self.entropy_lr = entropy_lr
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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
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
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        self.log_probs.append(log_prob)
        self.entrops.append(entropy)
        value = self.critic(state)
        self.values.append(value)
        return action.item()

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, episode_over: bool):
        self.rewards.append(reward)
        if episode_over:
            self.step_count = 0
            self.reward_history.append(np.sum(self.rewards))
            if len(self.rewards) > 1:
                self.episode_count += 1
                self.update_model()
                print(f"Episode: {self.episode_count} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")
            self.rewards.clear()

    def update_model(self):
        self.train_count += 1
        self.actor.train()
        g = np.array(self.rewards, dtype=np.float32)
        g /= self.reward_norm_factor
        r_sum = 0
        for i in reversed(range(g.shape[0])):
            g[i] = r_sum = r_sum * self.y + g[i]
        G = torch.tensor(g).unsqueeze(0).to(self.device)
        G -= G.mean()
        G /= (G.std() + self.eps)

        V = torch.cat(self.values, dim=1)
        # swapping position for no negative sign on actor_loss
        A = V.detach() - G

        LOG = torch.cat(self.log_probs)
        ENTROPY = torch.cat(self.entrops).mean()
        actor_loss = LOG @ A.T + self.entropy_lr * ENTROPY
        critic_loss = self.loss_fn(V, G)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.log_probs.clear()
        self.values.clear()
        self.entrops.clear()
