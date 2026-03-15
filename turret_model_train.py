import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from turret_env import TurretEnv


class PolicyGRU(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_low=None,
        action_high=None,
    ):
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        if action_low is None:
            action_low = [-np.pi, -np.pi / 3.0, 0.0]
        if action_high is None:
            action_high = [np.pi, np.pi / 3.0, 1.0]
        self._action_low_np = np.asarray(action_low, dtype=np.float32)
        self._action_high_np = np.asarray(action_high, dtype=np.float32)

    def _action_low(self, device):
        return torch.as_tensor(self._action_low_np, dtype=torch.float32, device=device)

    def _action_high(self, device):
        return torch.as_tensor(self._action_high_np, dtype=torch.float32, device=device)

    def _squash_to_action_space(self, raw_action):
        """
        Map unconstrained actions to simulator ranges with tanh instead of
        post-hoc clipping. This keeps training/inference behavior aligned.
        """
        low = self._action_low(raw_action.device)
        high = self._action_high(raw_action.device)
        action01 = 0.5 * (torch.tanh(raw_action) + 1.0)
        return low + (high - low) * action01

    def forward(self, obs_seq, hidden=None):
        x = torch.relu(self.obs_embed(obs_seq))
        out, h = self.gru(x, hidden)
        mu = self.mu(out)
        return mu, self.log_std.exp(), h

    def sample_action(self, obs_seq, hidden=None):
        mu, std, h = self.forward(obs_seq, hidden)
        dist = torch.distributions.Normal(mu, std)
        # REINFORCE / score-function gradient: do not backprop through the sampled action.
        # Using `rsample()` here would make `raw_action` a differentiable function of (mu, std)
        # and biases/cancels the intended ∇θ log πθ(a) update.
        raw_action = dist.sample()
        action = self._squash_to_action_space(raw_action)

        # Tanh squashing correction for log-probability.
        # log(1 - tanh(x)^2) is numerically stable with this formulation.
        log_det_jacobian = 2.0 * (
            np.log(2.0) - raw_action - torch.nn.functional.softplus(-2.0 * raw_action)
        )
        log_prob = dist.log_prob(raw_action).sum(-1) - log_det_jacobian.sum(-1)
        return action, log_prob, h

    def mean_action(self, obs_seq, hidden=None):
        """Deterministic action used during evaluation."""
        mu, _, h = self.forward(obs_seq, hidden)
        return self._squash_to_action_space(mu), h


def collect_episode(
    env: TurretEnv,
    policy: PolicyGRU,
    device,
    max_steps: int = 500,
    reset_seed: int | None = None,
):

    obs_buf = []
    act_buf = []
    logprob_buf = []
    reward_buf = []

    obs = env.reset(seed=reset_seed)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs = obs.unsqueeze(0).unsqueeze(0)

    done = False
    hidden = None

    for _ in range(max_steps):

        action, log_prob, hidden = policy.sample_action(obs, hidden)

        # `action` tracks gradients for policy optimization; env stepping should not.
        action_np = action.squeeze(0).squeeze(0).detach().cpu().numpy()

        obs_next, reward, done, _ = env.step(action_np)

        obs_next_t = torch.as_tensor(
            obs_next, dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)

        obs_buf.append(obs.squeeze(0))
        act_buf.append(action.squeeze(0))
        # Keep log-prob as a 1D tensor of length T so it aligns with returns.
        logprob_buf.append(log_prob.squeeze())
        reward_buf.append(reward)

        obs = obs_next_t

        if done:
            break

    traj = {
        "obs": torch.cat(obs_buf, dim=0),
        "act": torch.cat(act_buf, dim=0),
        "logprob": torch.stack(logprob_buf, dim=0).view(-1),
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


def evaluate_policy(env, policy, device, episodes=10, max_steps=500, base_seed: int | None = None):
    """
    Runs deterministic evaluation episodes and returns mean reward.
    """

    policy.eval()

    returns = []

    for _ in range(episodes):

        if base_seed is None:
            obs = env.reset()
        else:
            obs = env.reset(seed=base_seed + len(returns))
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        obs = obs.unsqueeze(0).unsqueeze(0)

        done = False
        hidden = None
        total_reward = 0

        for _ in range(max_steps):

            with torch.no_grad():
                action, hidden = policy.mean_action(obs, hidden)

            action_np = action.squeeze(0).squeeze(0).detach().cpu().numpy()

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


def train(
    n_episodes: int = 100000,
    eval_interval: int = 50,
    eval_episodes: int = 25,
    lr: float = 12e-4,
    grad_clip_norm: float = 1.0,
    seed: int | None = None,
    eval_seed: int | None = 12345,
    save_current_every: int = 10,
):

    # Each step, the simulator computes a non-ML baseline aim (yaw/pitch-to-panel),
    # and the policy outputs a correction on top of it.
    env = TurretEnv(action_is_correction=True, correction_baseline="panel", seed=seed)

    obs_dim = 7
    action_dim = 3
    hidden_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # In correction mode, actions are per-step corrections, not absolute yaw/pitch.
    # Keep their range modest; large ranges let the policy "learn" ~pi (180°) flips.
    correction_yaw_max = float(getattr(env, "correction_clip_yaw", np.deg2rad(30.0)) or np.deg2rad(30.0))
    correction_pitch_max = float(getattr(env, "correction_clip_pitch", np.deg2rad(15.0)) or np.deg2rad(15.0))
    policy = PolicyGRU(
        obs_dim,
        hidden_dim,
        action_dim,
        action_low=[-correction_yaw_max, -correction_pitch_max, 0.0],
        action_high=[correction_yaw_max, correction_pitch_max, 1.0],
    ).to(device)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    optimizer = optim.Adam(policy.parameters(), lr=float(lr))

    best_return = -np.inf

    best_model_path = "best_policy_gru.pt"
    current_model_path = "current_gru.pt"

    for episode in range(1, n_episodes + 1):

        traj = collect_episode(
            env,
            policy,
            device,
            reset_seed=None if seed is None else int(seed + episode),
        )

        returns = compute_returns(traj["reward"], gamma=0.99).detach()

        # Robust normalization: avoid NaNs for very short episodes.
        if returns.numel() > 1:
            ret_std = returns.std(unbiased=False)
        else:
            ret_std = torch.tensor(1.0, device=returns.device, dtype=returns.dtype)
        returns = (returns - returns.mean()) / (ret_std + 1e-8)

        log_probs = traj["logprob"]
        if log_probs.numel() != returns.numel():
            raise RuntimeError(
                f"log_probs and returns must align; got log_probs={tuple(log_probs.shape)} returns={tuple(returns.shape)}"
            )

        # Use mean loss so gradient scale doesn't grow with episode length.
        loss = -(log_probs.view(-1) * returns.view(-1)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at episode {episode}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

        if save_current_every and (episode % int(save_current_every) == 0):
            torch.save(policy.state_dict(), current_model_path)

        total_return = traj["reward"].sum().item()

        if episode % 10 == 0:
            print(
                f"Episode {episode}, Return={total_return:.1f}, Length={len(traj['reward'])}, BestEval={best_return:.1f}"
            )

        if episode % eval_interval == 0:

            eval_return = evaluate_policy(
                env,
                policy,
                device,
                episodes=eval_episodes,
                base_seed=None if eval_seed is None else int(eval_seed),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--lr", type=float, default=12e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--save-current-every", type=int, default=10)
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        lr=args.lr,
        grad_clip_norm=args.grad_clip,
        seed=args.seed,
        eval_seed=args.eval_seed,
        save_current_every=args.save_current_every,
    )
