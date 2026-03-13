"""
Simple PyTorch policy-gradient example on CartPole-v1.

This trains a small MLP policy to balance the pole.
It is intentionally compact so you can adapt it later
for your turret environment or a GRU-based policy.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)


def run_episode(env, policy: PolicyNet, device: torch.device):
    """Run one episode and collect log-probs and rewards."""
    log_probs = []
    rewards = []

    obs, _ = env.reset()
    done = False

    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = policy(obs_t)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        obs, reward, terminated, truncated, _ = env.step(action.item())

        log_probs.append(log_prob)
        rewards.append(reward)

        done = terminated or truncated

    return log_probs, rewards


def compute_returns(rewards, gamma: float):
    """Compute discounted returns for a list of rewards."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def main():
    env = gym.make("CartPole-v1", render_mode=None)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, hidden_dim=64, n_actions=n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    gamma = 0.99
    n_episodes = 10000

    for episode in range(n_episodes):
        log_probs, rewards = run_episode(env, policy, device)
        returns = compute_returns(rewards, gamma)

        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        # Normalize returns to stabilize training
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * returns_t).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = sum(rewards)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}, return = {episode_return:.1f}")

    env.close()


if __name__ == "__main__":
    main()
