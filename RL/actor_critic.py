import torch
from torch.distributions import Categorical
import numpy as np
from .deep_agent import DeepAgent


class ActorCriticAgent(DeepAgent):

    def __init__(self, state_space_size: int, action_space_size: int, device: str = 'cpu') -> None:
        super().__init__(state_space_size, action_space_size, device)
        self.actor = None
        self.critic = None
        self.eps = np.finfo(np.float32).eps.item()
        self.loss_fn = torch.nn.HuberLoss(reduction='mean')
        self.reward_norm_factor = 1.0
        self.gae_lambda = 1.0
        self.entropy_coef = 0.1
        self.env_count = 1
        self.step_count = 1
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
                     gae_lambda: float,
                     env_count: int,
                     step_count: int,
                     reward_norm_factor: float = 1.0):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.env_count = env_count
        self.step_count = step_count
        self.reward_norm_factor = reward_norm_factor
        self.actor = actor(self.state_space_size, self.action_space_size)
        self.actor.to(self.device)
        self.actor.train()
        self.critic = critic(self.state_space_size)
        self.critic.to(self.device)
        self.critic.train()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.state_buffer = np.zeros((env_count, step_count, self.state_space_size))
        self.next_state_buffer = np.zeros((env_count, step_count, self.state_space_size))
        self.action_buffer = np.zeros((env_count, step_count))
        self.reward_buffer = np.zeros((env_count, step_count))
        self.done_buffer = np.zeros((env_count, step_count))

    @torch.no_grad()
    def policy(self, state: np.ndarray):
        self.step_counter += 1
        self.actor.eval()
        if state.ndim == 1:
            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        else:
            state = torch.tensor(state).float().to(self.device)
        probs = self.actor(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action.cpu().numpy()

    def learn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: np.ndarray, done: np.ndarray):
        self.rewards.append(np.mean(reward))
        self.state_buffer[:, self.step_counter - 1] = state
        self.action_buffer[:, self.step_counter - 1] = action
        self.next_state_buffer[:, self.step_counter - 1] = next_state
        self.reward_buffer[:, self.step_counter - 1] = reward
        self.done_buffer[:, self.step_counter - 1] = done
        if any(done):
            self.update_model()
            self.state_buffer = np.zeros_like(self.state_buffer)
            self.next_state_buffer = np.zeros_like(self.next_state_buffer)
            self.action_buffer = np.zeros_like(self.action_buffer)
            self.reward_buffer = np.zeros_like(self.reward_buffer)
            self.done_buffer = np.zeros_like(self.done_buffer)
            self.step_counter = 0
            self.episode_counter += 1
            self.reward_history.append(np.sum(self.rewards))
            self.rewards.clear()
            print(f"Episode: {self.episode_counter} | Train: {self.train_count} | r: {self.reward_history[-1]:.6f}")

    def update_model(self):
        self.train_count += 1
        self.actor.train()

        states = torch.tensor(self.state_buffer[:, :self.step_counter]).float().to(self.device)
        actions = torch.tensor(self.action_buffer[:, :self.step_counter]).to(self.device)
        next_states = torch.tensor(self.next_state_buffer[:, :self.step_counter]).float().to(self.device)
        rewards = torch.tensor(self.reward_buffer[:, :self.step_counter]).float().to(self.device)
        dones = 1 - torch.tensor(self.done_buffer[:, :self.step_counter]).to(self.device)

        rewards /= self.reward_norm_factor

        actor_losses = []
        critic_losses = []

        for i in range(self.env_count):
            probs = self.actor(states[i])
            dist = Categorical(probs=probs)
            LOG = dist.log_prob(actions[i])
            V = self.critic(states[i]).view(-1)
            A = self.GAE(next_states[i], rewards[i], V, dones[i])
            G = A + V.detach()

            actor_loss = (LOG * -A).mean()
            critic_loss = self.loss_fn(V, G)

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    @torch.no_grad()
    def GAE(self, next_states, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        next_value = 0
        next_done = 1
        for i in reversed(range(self.step_counter)):
            if i == self.step_counter - 1:
                next_done = dones[i]
                next_value = self.critic(next_states[i])
            else:
                next_done = dones[i + 1]
                next_value = values[i + 1]
            current_value = values[i]
            delta = rewards[i] + self.gamma * next_value * next_done - current_value
            advantages[i] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * next_done
        return advantages
