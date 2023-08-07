import torch
from torch.distributions import Categorical
import numpy as np
from .deep_agent import DeepAgent


class ProximalPolicyOptimizationAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.actor = None
        self.critic = None
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss(reduction='mean')
        self.reward_norm_factor = 1.0
        self.gae_lambda = 1.0
        self.entropy_coef = 0.1
        self.clip_coef = 0.2
        self.kl_threshold = 0.02
        self.step_count = 0
        self.batch = 0
        self.epoch = 0
        del self.model
        del self.optimizer
        del self.lr

    def create_model(self,
                     actor: torch.nn.Module,
                     critic: torch.nn.Module,
                     actor_lr: float,
                     critic_lr: float,
                     gamma: float,
                     entropy_coef: float,
                     clip_coef: float,
                     kl_threshold: float,
                     gae_lambda: float,
                     step_count: int,
                     batch: int,
                     epoch: int,
                     reward_norm_factor: float = 1.0):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.clip_coef = clip_coef
        self.kl_threshold = kl_threshold
        self.step_count = step_count
        self.batch = batch
        self.epoch = epoch
        self.reward_norm_factor = reward_norm_factor
        self.actor = actor(self.state_space_size, self.action_space_size)
        self.actor.to(self.device)
        self.actor.train()
        self.critic = critic(self.state_space_size)
        self.critic.to(self.device)
        self.critic.train()
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.log_prob_buffer = []
        self.value_buffer = []

    def policy(self, state: np.ndarray):
        self.step_counter += 1
        if state.ndim == 1:
            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        else:
            state = torch.tensor(state).float().to(self.device)

        if not self.training:
            self.actor.eval()
        else:
            self.actor.train()

        with torch.no_grad():
            probs = self.actor(state).squeeze(0)
            distribution = Categorical(probs)
            action = distribution.sample()

        if not self.training:
            return int(action.cpu().numpy())

        with torch.no_grad():
            log_prob = distribution.log_prob(action)
            value = self.critic(state).squeeze(0)
        self.log_prob_buffer.append(log_prob)
        self.value_buffer.append(value)
        return int(action.cpu().numpy())

    def learn(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool):
        self.rewards.append(reward)
        self.state_buffer.append(np.copy(state))
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        if done:
            self.step_counter = 0
            self.episode_counter += 1
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_counter} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")
        if len(self.done_buffer) == self.step_count:
            self.update_model(next_state)
            self.state_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()
            self.done_buffer.clear()
            self.log_prob_buffer.clear()
            self.value_buffer.clear()

    def update_model(self, last_state):
        self.train_count += 1

        b_states = torch.tensor(np.array(self.state_buffer)).float().to(self.device)
        b_actions = torch.tensor(self.action_buffer).to(self.device)
        b_rewards = torch.tensor(self.reward_buffer).float().to(self.device)
        b_dones = 1 - torch.tensor(self.done_buffer).long().to(self.device)
        b_rewards /= self.reward_norm_factor
        b_log_probs = torch.stack(self.log_prob_buffer).to(self.device).view(-1)
        b_values = torch.stack(self.value_buffer).to(self.device).view(-1)
        b_advantage, b_return = self.GAE(last_state, b_dones, b_rewards, b_values)

        b_inds = np.arange(self.step_count)

        for _ in range(self.epoch):
            if self.step_count <= self.batch:
                mb_inds = b_inds
            else:
                mb_inds = np.random.choice(b_inds, self.batch, False)

            distribution = Categorical(probs=self.actor(b_states[mb_inds]))
            LOG = distribution.log_prob(b_actions[mb_inds])

            ENTROPY = distribution.entropy()

            log_ratio = LOG - b_log_probs[mb_inds]
            RATIO = log_ratio.exp()

            V = self.critic(b_states[mb_inds]).view(-1)

            with torch.no_grad():
                approx_kl = ((RATIO - 1) - log_ratio).mean()

            mb_advantage = b_advantage[mb_inds]
            actor_loss1 = mb_advantage * RATIO
            actor_loss2 = mb_advantage * torch.clamp(RATIO, 1 - self.clip_coef, 1 + self.clip_coef)
            actor_loss = -torch.min(actor_loss1, actor_loss2).mean()

            entropy_loss = ENTROPY.mean()

            actor_loss = actor_loss - entropy_loss * self.entropy_coef

            critic_loss = self.loss_fn(V, b_return[mb_inds])

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if approx_kl >= self.kl_threshold:
                print('break')
                break

    @torch.no_grad()
    def GAE(self, last_state, dones, rewards, values):
        advantages = torch.zeros_like(rewards).to(self.device)
        future_value = self.critic(torch.from_numpy(last_state).float().to(self.device)) * dones[self.step_count - 1]
        last_advantage = 0
        for i in reversed(range(self.step_count)):
            if not dones[i]:
                if i == self.step_count - 1:
                    pass
                else:
                    future_value = 0
                    last_advantage = 0
            else:
                if i == self.step_count - 1:
                    pass
                else:
                    future_value = values[i + 1]
            delta = rewards[i] + self.gamma * future_value - values[i]
            advantages[i] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
        returns = advantages + values
        return advantages, returns

    @torch.no_grad()
    def VAE(self, last_state, rewards, values, done):
        returns = torch.zeros_like(rewards)
        r_sum = self.critic(last_state) * done
        for i in reversed(range(self.step_counter)):
            returns[i] = r_sum = r_sum * self.gamma + rewards[i]
        advantages = returns - values
        return advantages, returns
