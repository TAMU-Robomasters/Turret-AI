import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from turret_env import TurretEnv


class PolicyGRU(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs_seq, hidden=None):
        x = torch.relu(self.obs_embed(obs_seq))
        out, h = self.gru(x, hidden)
        mu = self.mu(out)
        return mu, self.log_std.exp(), h

    def sample_action(self, obs_seq, hidden=None):
        mu, std, h = self.forward(obs_seq, hidden)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, h


def collect_episode(env: TurretEnv, policy: PolicyGRU, device, max_steps=500):

    obs_buf = []
    act_buf = []
    logprob_buf = []
    reward_buf = []

    obs = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs = obs.unsqueeze(0).unsqueeze(0)

    done = False
    hidden = None

    for _ in range(max_steps):

        action, log_prob, hidden = policy.sample_action(obs, hidden)

        action_np = action.squeeze(0).squeeze(0).cpu().numpy()

        action_np[0] = np.clip(action_np[0], -np.pi, np.pi)
        action_np[1] = np.clip(action_np[1], -np.pi/3, np.pi/3)
        action_np[2] = np.clip(action_np[2], 0, 1.0)

        obs_next, reward, done, _ = env.step(action_np)

        obs_next_t = torch.as_tensor(
            obs_next, dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)

        obs_buf.append(obs.squeeze(0))
        act_buf.append(action.squeeze(0))
        logprob_buf.append(log_prob.squeeze(0))
        reward_buf.append(reward)

        obs = obs_next_t

        if done:
            break

    traj = {
        "obs": torch.cat(obs_buf, dim=0),
        "act": torch.cat(act_buf, dim=0),
        "logprob": torch.stack(logprob_buf, dim=0),
        "reward": torch.as_tensor(reward_buf, dtype=torch.float32, device=device),
    }

    return traj


def compute_returns(rewards, gamma=0.99):

    returns = []
    G = 0.0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return torch.as_tensor(returns, dtype=torch.float32, device=rewards.device)


def evaluate_policy(env, policy, device, episodes=10, max_steps=500):
    """
    Runs deterministic evaluation episodes and returns mean reward.
    """

    policy.eval()

    returns = []

    for _ in range(episodes):

        obs = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        obs = obs.unsqueeze(0).unsqueeze(0)

        done = False
        hidden = None
        total_reward = 0

        for _ in range(max_steps):

            with torch.no_grad():
                mu, _, hidden = policy(obs, hidden)
                action = mu

            action_np = action.squeeze(0).squeeze(0).cpu().numpy()

            action_np[0] = np.clip(action_np[0], -np.pi, np.pi)
            action_np[1] = np.clip(action_np[1], -np.pi/3, np.pi/3)
            action_np[2] = np.clip(action_np[2], 0, 1.0)

            obs, reward, done, _ = env.step(action_np)

            obs = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0)

            total_reward += reward

            if done:
                break

        returns.append(total_reward)

    policy.train()

    return np.mean(returns)


def train():

    env = TurretEnv()

    obs_dim = 7
    action_dim = 3
    hidden_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyGRU(obs_dim, hidden_dim, action_dim).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    best_return = -np.inf

    best_model_path = "best_policy_gru.pt"
    current_model_path = "current_gru.pt"

    n_episodes = 100000
    eval_interval = 50
    eval_episodes = 10

    for episode in range(1, n_episodes + 1):

        traj = collect_episode(env, policy, device)

        returns = compute_returns(traj["reward"], gamma=0.99).detach()

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = traj["logprob"]

        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(policy.state_dict(), current_model_path)

        total_return = traj["reward"].sum().item()

        if episode % 10 == 0:
            print(
                f"Episode {episode}, Return={total_return:.1f}, Length={len(traj['reward'])}, BestEval={best_return:.1f}"
            )

        if episode % eval_interval == 0:

            eval_return = evaluate_policy(
                env, policy, device, episodes=eval_episodes
            )

            print(
                f"Evaluation after {episode} episodes -> Mean Return: {eval_return:.2f}"
            )

            if eval_return > best_return:

                best_return = eval_return

                torch.save(policy.state_dict(), best_model_path)

                print(
                    f"New best model! Eval Return={eval_return:.2f}"
                )


if __name__ == "__main__":
    train()