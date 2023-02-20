import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class PG(torch.nn.Module):

    def __init__(self, observation_size=4, action_size=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(observation_size, 16)
        self.fc2 = torch.nn.Linear(16, action_size)
        self.log_probs = []

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=0)
        m = Categorical(x)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()


Y = 0.9
LR = 0.01

env = gym.make("CartPole-v1", render_mode="human")
policy = PG()
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

observation, info = env.reset(seed=42)
rewards = []
train_count = 0
while True:
    obs = torch.from_numpy(observation)
    action = policy(obs)  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)

    if terminated or truncated:
        train_count += 1
        observation, info = env.reset()
        last = 0
        discounted_returns = []
        for item in reversed(rewards):
            last = (last * Y + item)
            discounted_returns.append(last)
        returns = torch.tensor(list(reversed(discounted_returns)))
        # print(returns)
        returns -= returns.mean()
        # print(returns)
        if len(returns) > 1:
            returns /= (returns.std())
        loss = torch.tensor([-log_prob * discounted_return
                             for log_prob, discounted_return
                             in zip(policy.log_probs, returns)],
                            requires_grad=True).sum()

        # if self.train_count % 10 == 0:
        print(f"Train: {train_count} - loss --------------------------> {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards = []
        policy.log_probs = []
env.close()
