import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import turret_sim
from turret_env import TurretEnv


class PolicyGRU(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 1,
        action_low=None,
        action_high=None,
    ):
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=int(num_layers), batch_first=True)
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



class ValueGRU(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=int(num_layers), batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """
        obs_seq: (T, obs_dim) or (B, T, obs_dim)
        returns: (T,) or (B, T)
        """
        squeeze_output = False
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
            squeeze_output = True

        x = torch.relu(self.obs_embed(obs_seq))
        out, _ = self.gru(x, hidden)
        values = self.value_head(out).squeeze(-1)
        if squeeze_output:
            values = values.squeeze(0)
        return values


class TrainingLogger:
    """CSV logger for plotting training curves."""

    def __init__(self, log_dir: str | None = "checkpoints"):
        self.log_dir = log_dir
        self.log_file = None
        self._header_written = False
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(os.path.join(log_dir, "training.log"), "w")

    def _write_header(self, columns: list[str]) -> None:
        if self.log_file is None or self._header_written:
            return
        self.log_file.write(",".join(columns) + "\n")
        self.log_file.flush()
        self._header_written = True

    def write_row(self, row: dict) -> None:
        if self.log_file is None:
            return
        columns = list(row.keys())
        self._write_header(columns)
        values = []
        for key in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        self.log_file.write(",".join(values) + "\n")
        self.log_file.flush()

    def close(self) -> None:
        if self.log_file is not None:
            self.log_file.close()


def _raw_action_from_squashed(policy: nn.Module, action: torch.Tensor) -> torch.Tensor:
    """
    Invert policy's tanh+affine action squash:
      action = low + (high-low) * 0.5 * (tanh(raw) + 1)
    """
    low = torch.as_tensor(policy._action_low_np, dtype=action.dtype, device=action.device)
    high = torch.as_tensor(policy._action_high_np, dtype=action.dtype, device=action.device)
    denom = torch.clamp(high - low, min=1e-6)
    action01 = torch.clamp((action - low) / denom, 1e-6, 1.0 - 1e-6)
    tanh_raw = 2.0 * action01 - 1.0
    tanh_raw = torch.clamp(tanh_raw, -1.0 + 1e-6, 1.0 - 1e-6)
    return 0.5 * (torch.log1p(tanh_raw) - torch.log1p(-tanh_raw))


def ppo_logprob_entropy_from_traj(policy: nn.Module, obs_traj: torch.Tensor, act_traj: torch.Tensor):
    """
    Recompute log-probabilities and entropy for a trajectory in temporal order.
    obs_traj: (T, obs_dim), act_traj: (T, action_dim)
    """
    hidden = None
    logprob_list = []
    entropy_list = []
    for t in range(obs_traj.shape[0]):
        obs_t = obs_traj[t].unsqueeze(0).unsqueeze(0)
        mu, std, hidden = policy.forward(obs_t, hidden)
        mu = mu.squeeze(0).squeeze(0)
        dist = torch.distributions.Normal(mu, std)

        act_t = act_traj[t]
        raw_t = _raw_action_from_squashed(policy, act_t)
        log_det_jacobian = 2.0 * (
            np.log(2.0) - raw_t - torch.nn.functional.softplus(-2.0 * raw_t)
        )
        log_prob_t = dist.log_prob(raw_t).sum(-1) - log_det_jacobian.sum(-1)
        entropy_t = dist.entropy().sum(-1)

        logprob_list.append(log_prob_t)
        entropy_list.append(entropy_t)

    return torch.stack(logprob_list), torch.stack(entropy_list)


def collect_episode(
    env: TurretEnv,
    policy: nn.Module,
    device,
    max_steps: int = 500,
    reset_seed: int | None = None,
):

    obs_buf = []
    act_buf = []
    logprob_buf = []
    reward_buf = []
    reward_parts_sum = {
        "hit": 0.0,
        "shot": 0.0,
        "align": 0.0,
        "track": 0.0,
    }

    k = int(getattr(policy, "obs_history_k", 1) or 1)
    obs_np = env.reset(seed=reset_seed)
    if k > 1:
        obs_hist = [np.asarray(obs_np, dtype=np.float32).copy() for _ in range(k)]
        obs_stack = np.concatenate(obs_hist, axis=0)
        obs = torch.as_tensor(obs_stack, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    else:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    done = False
    hidden = None

    for _ in range(max_steps):

        action, log_prob, hidden = policy.sample_action(obs, hidden)

        # `action` tracks gradients for policy optimization; env stepping should not.
        action_np = action.squeeze(0).squeeze(0).detach().cpu().numpy()

        obs_next, reward, done, info = env.step(action_np)
        if isinstance(info, dict) and "reward_parts" in info:
            for part_key, part_val in info["reward_parts"].items():
                if part_key in reward_parts_sum:
                    reward_parts_sum[part_key] += float(part_val)

        if k > 1:
            obs_hist.pop(0)
            obs_hist.append(np.asarray(obs_next, dtype=np.float32))
            obs_stack_next = np.concatenate(obs_hist, axis=0)
            obs_next_t = torch.as_tensor(obs_stack_next, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        else:
            obs_next_t = torch.as_tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

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
        "terminal": bool(done),
        "reward_parts": reward_parts_sum,
    }

    return traj


def compute_returns(rewards, gamma=0.99):

    returns = []
    G = 0.0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return torch.as_tensor(returns, dtype=torch.float32, device=rewards.device)


def evaluate_policy(env, policy: nn.Module, device, episodes=10, max_steps=500, base_seed: int | None = None):
    """
    Runs deterministic evaluation episodes and returns mean reward.
    """

    policy.eval()

    returns = []
    k = int(getattr(policy, "obs_history_k", 1) or 1)

    for _ in range(episodes):

        if base_seed is None:
            obs = env.reset()
        else:
            obs = env.reset(seed=base_seed + len(returns))
        if k > 1:
            obs_hist = [np.asarray(obs, dtype=np.float32).copy() for _ in range(k)]
            obs_stack = np.concatenate(obs_hist, axis=0)
            obs = torch.as_tensor(obs_stack, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        done = False
        hidden = None
        total_reward = 0

        for _ in range(max_steps):

            with torch.no_grad():
                action, hidden = policy.mean_action(obs, hidden)

            action_np = action.squeeze(0).squeeze(0).detach().cpu().numpy()

            obs, reward, done, _ = env.step(action_np)

            if k > 1:
                obs_hist.pop(0)
                obs_hist.append(np.asarray(obs, dtype=np.float32))
                obs_stack_next = np.concatenate(obs_hist, axis=0)
                obs = torch.as_tensor(obs_stack_next, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            else:
                obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

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
    lr: float = 3e-4,
    grad_clip_norm: float = 1.0,
    seed: int | None = None,
    eval_seed: int | None = 12345,
    save_current_every: int = 10,
    shot_base_penalty: float = 10.0,
    shot_miss_penalty_scale: float = 20.0,
    shot_miss_target_radius_mm: float = 60.0,
    shot_miss_penalty_power: float = 1.0,
    shot_miss_max_ratio: float = 5.0,
    shot_miss_max_angle_deg: float = 45.0,
    hidden_dim: int = 256,
    num_layers: int = 1,
    debug_interval: int = 10,
    debug_window: int = 50,
    policy_std_decay: float = 0.9995,
    policy_std_min: float = 0.15,
    eval_patience: int = 12,
    eval_min_delta: float = 1.0,
    algo: str = "reinforce",
    ppo_epochs: int = 4,
    ppo_clip_coef: float = 0.2,
    ppo_value_coef: float = 0.5,
    ppo_entropy_coef: float = 0.01,
    log_dir: str = "checkpoints",
):
    # Disable simulator debug visuals/logs during training.
    # This does not affect the training progress prints in this file.
    turret_sim.DEBUG = False

    # Each step, the simulator computes a non-ML baseline aim (yaw/pitch-to-panel),
    # and the policy outputs a correction on top of it.
    env = TurretEnv(
        action_is_correction=True,
        correction_baseline="panel",
        shot_base_penalty=shot_base_penalty,
        shot_miss_penalty_scale=shot_miss_penalty_scale,
        shot_miss_target_radius_mm=shot_miss_target_radius_mm,
        shot_miss_penalty_power=shot_miss_penalty_power,
        shot_miss_max_ratio=shot_miss_max_ratio,
        shot_miss_max_angle_deg=shot_miss_max_angle_deg,
        seed=seed,
    )

    base_obs_dim = 4
    action_dim = 3
    hidden_dim = int(hidden_dim)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Check your NVIDIA driver/CUDA setup.")
    device = torch.device("cuda")
    print(device)
    logger = TrainingLogger(log_dir=log_dir)

    # In correction mode, actions are per-step corrections, not absolute yaw/pitch.
    # Keep their range modest; large ranges let the policy "learn" ~pi (180°) flips.
    correction_yaw_max = float(getattr(env, "correction_clip_yaw", np.deg2rad(30.0)) or np.deg2rad(30.0))
    correction_pitch_max = float(getattr(env, "correction_clip_pitch", np.deg2rad(15.0)) or np.deg2rad(15.0))
    action_low = [-correction_yaw_max, -correction_pitch_max, 0.0]
    action_high = [correction_yaw_max, correction_pitch_max, 1.0]

    algo = str(algo).lower().strip()
    obs_dim = base_obs_dim
    policy = PolicyGRU(
        obs_dim,
        hidden_dim,
        action_dim,
        num_layers=int(num_layers),
        action_low=action_low,
        action_high=action_high,
    ).to(device)
    best_model_path = "best_policy_gru.pt"
    current_model_path = "current_gru.pt"

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    optimizer = optim.Adam(policy.parameters(), lr=float(lr))
    value_net = None
    value_optimizer = None
    if algo == "ppo":
        value_net = ValueGRU(obs_dim=obs_dim, hidden_dim=hidden_dim, num_layers=int(num_layers)).to(device)
        value_optimizer = optim.Adam(value_net.parameters(), lr=float(lr))
    elif algo != "reinforce":
        raise ValueError("Unknown --algo %r (expected 'reinforce' or 'ppo')." % (algo,))

    best_return = -np.inf
    eval_no_improve = 0
    recent_returns = deque(maxlen=max(1, int(debug_window)))
    recent_lengths = deque(maxlen=max(1, int(debug_window)))
    recent_losses = deque(maxlen=max(1, int(debug_window)))
    recent_eval = deque(maxlen=max(1, int(debug_window)))

    for episode in range(1, n_episodes + 1):

        traj = collect_episode(
            env,
            policy,
            device,
            reset_seed=None if seed is None else int(seed + episode),
        )

        grad_norm = 0.0
        if algo == "reinforce":
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
            grad_norm_sq = 0.0
            for p in policy.parameters():
                if p.grad is not None:
                    grad_norm_sq += float(p.grad.detach().norm(2).item() ** 2)
            grad_norm = float(np.sqrt(grad_norm_sq))
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            loss_item = float(loss.item())
        else:
            obs_traj = traj["obs"].detach()
            act_traj = traj["act"].detach()
            old_logprob = traj["logprob"].detach()
            returns = compute_returns(traj["reward"], gamma=0.99).detach()

            with torch.no_grad():
                values_old = value_net(obs_traj)
                advantages = returns - values_old
                adv_std = advantages.std(unbiased=False) if advantages.numel() > 1 else torch.tensor(1.0, device=device)
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            policy_losses = []
            value_losses = []
            entropy_vals = []

            for ppo_epoch_idx in range(max(1, int(ppo_epochs))):
                new_logprob, entropy = ppo_logprob_entropy_from_traj(policy, obs_traj, act_traj)
                if episode == 1 and ppo_epoch_idx == 0:
                    # Sanity check: recomputing log-probabilities from stored
                    # (squashed) actions should closely match `old_logprob`.
                    with torch.no_grad():
                        logprob_diff = float((new_logprob - old_logprob).abs().mean().item())
                        ratio_mean = float(torch.exp(new_logprob - old_logprob).mean().item())
                    print(
                        f"PPO logprob sanity: mean|new-old|={logprob_diff:.6f}, ratio_mean={ratio_mean:.4f}"
                    )
                ratio = torch.exp(new_logprob - old_logprob)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - float(ppo_clip_coef),
                    1.0 + float(ppo_clip_coef),
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = value_net(obs_traj)
                value_loss = torch.nn.functional.mse_loss(value_pred, returns)
                entropy_bonus = entropy.mean()

                total_loss = (
                    policy_loss
                    + float(ppo_value_coef) * value_loss
                    - float(ppo_entropy_coef) * entropy_bonus
                )

                optimizer.zero_grad(set_to_none=True)
                value_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()

                grad_norm_sq = 0.0
                for p in policy.parameters():
                    if p.grad is not None:
                        grad_norm_sq += float(p.grad.detach().norm(2).item() ** 2)
                grad_norm = float(np.sqrt(grad_norm_sq))

                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(grad_clip_norm))
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=float(grad_clip_norm))

                optimizer.step()
                value_optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_vals.append(float(entropy_bonus.item()))

            loss_item = float(np.mean(policy_losses)) if policy_losses else 0.0
        if hasattr(policy, "log_std") and isinstance(policy.log_std, torch.nn.Parameter):
            # Gradually reduce exploration while keeping a floor so policy can adapt.
            with torch.no_grad():
                std_now = policy.log_std.exp()
                std_next = torch.clamp(
                    std_now * float(policy_std_decay),
                    min=float(policy_std_min),
                )
                policy.log_std.copy_(std_next.log())

        if save_current_every and (episode % int(save_current_every) == 0):
            torch.save(policy.state_dict(), current_model_path)

        total_return = traj["reward"].sum().item()
        ep_len = int(len(traj["reward"]))
        recent_returns.append(float(total_return))
        recent_lengths.append(ep_len)
        recent_losses.append(float(loss_item))

        if episode % max(1, int(debug_interval)) == 0:
            avg_return = float(np.mean(recent_returns)) if recent_returns else float(total_return)
            avg_len = float(np.mean(recent_lengths)) if recent_lengths else float(ep_len)
            avg_loss = float(np.mean(recent_losses)) if recent_losses else float(loss_item)
            policy_std = float(policy.log_std.detach().exp().mean().item())
            eval_avg = float(np.mean(recent_eval)) if recent_eval else float("nan")
            eval_part = f", EvalAvg({len(recent_eval)})={eval_avg:.1f}" if recent_eval else ""
            msg = (
                f"Episode {episode}, Return={total_return:.1f}, Len={ep_len}, "
                f"AvgReturn({len(recent_returns)})={avg_return:.1f}, AvgLen({len(recent_lengths)})={avg_len:.1f}, "
                f"AvgLoss({len(recent_losses)})={avg_loss:.4f}, GradNorm={grad_norm:.3f}, Algo={algo}, "
                f"PolicyStd={policy_std:.3f}, BestEval={best_return:.1f}{eval_part}"
            )
            print(msg)
            logger.write_row(
                {
                    "type": "train",
                    "episode": episode,
                    "return": total_return,
                    "length": ep_len,
                    "avg_return": avg_return,
                    "avg_length": avg_len,
                    "avg_loss": avg_loss,
                    "grad_norm": grad_norm,
                    "policy_std": policy_std,
                    "best_eval": best_return,
                    "eval_avg": eval_avg,
                    "hit": float(traj["reward_parts"]["hit"]),
                    "shot": float(traj["reward_parts"]["shot"]),
                    "align": float(traj["reward_parts"]["align"]),
                    "track": float(traj["reward_parts"]["track"]),
                }
            )
        if episode % 10 == 0:
            rp = traj["reward_parts"]
            msg = (
                f"ep={episode} ret={total_return:.1f} len={len(traj['reward'])} "
                f"hit={rp['hit']:.1f} shot={rp['shot']:.1f} "
                f"align={rp['align']:.1f} track={rp['track']:.1f}"
            )
            print(msg)
            logger.write_row(
                {
                    "type": "reward_parts",
                    "episode": episode,
                    "return": total_return,
                    "length": ep_len,
                    "hit": float(rp["hit"]),
                    "shot": float(rp["shot"]),
                    "align": float(rp["align"]),
                    "track": float(rp["track"]),
                }
            )

        if episode % eval_interval == 0:

            eval_return = evaluate_policy(
                env,
                policy,
                device,
                episodes=eval_episodes,
                base_seed=None if eval_seed is None else int(eval_seed),
            )

            eval_msg = f"Evaluation after {episode} episodes -> Mean Return: {eval_return:.2f}"
            print(eval_msg)
            logger.write_row(
                {
                    "type": "eval",
                    "episode": episode,
                    "eval_return": eval_return,
                }
            )
            recent_eval.append(float(eval_return))

            improved = eval_return > (best_return + float(eval_min_delta))
            if improved:

                best_return = eval_return
                eval_no_improve = 0

                torch.save(policy.state_dict(), best_model_path)

                msg = f"New best model! Eval Return={eval_return:.2f}"
                print(msg)
                logger.write_row(
                    {
                        "type": "eval_best",
                        "episode": episode,
                        "eval_return": eval_return,
                    }
                )
            else:
                eval_no_improve += 1
                msg = (
                    f"No eval improvement for {eval_no_improve}/{int(eval_patience)} checks "
                    f"(best={best_return:.2f}, current={eval_return:.2f})"
                )
                print(msg)
                logger.write_row(
                    {
                        "type": "eval_no_improve",
                        "episode": episode,
                        "eval_return": eval_return,
                        "best_eval": best_return,
                        "no_improve": eval_no_improve,
                    }
                )
                if int(eval_patience) > 0 and eval_no_improve >= int(eval_patience):
                    msg = (
                        "Early stopping: eval has not improved for "
                        f"{int(eval_patience)} checks (min_delta={float(eval_min_delta):.2f})."
                    )
                    print(msg)
                    logger.write_row(
                        {
                            "type": "early_stop",
                            "episode": episode,
                            "best_eval": best_return,
                            "no_improve": eval_no_improve,
                        }
                    )
                    break
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--save-current-every", type=int, default=10)
    parser.add_argument("--algo", type=str, default="ppo", choices=["reinforce", "ppo"])
    parser.add_argument("--shot-base-penalty", type=float, default=10.0)
    parser.add_argument("--shot-miss-penalty-scale", type=float, default=20.0)
    parser.add_argument("--shot-miss-target-radius-mm", type=float, default=60.0)
    parser.add_argument("--shot-miss-penalty-power", type=float, default=1.0)
    parser.add_argument("--shot-miss-max-ratio", type=float, default=5.0)
    parser.add_argument("--shot-miss-max-angle-deg", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--debug-interval", type=int, default=10)
    parser.add_argument("--debug-window", type=int, default=50)
    parser.add_argument("--policy-std-decay", type=float, default=0.9995)
    parser.add_argument("--policy-std-min", type=float, default=0.15)
    parser.add_argument("--eval-patience", type=int, default=12)
    parser.add_argument("--eval-min-delta", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip-coef", type=float, default=0.2)
    parser.add_argument("--ppo-value-coef", type=float, default=0.5)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-dir", type=str, default="checkpoints")
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
        algo=args.algo,
        shot_base_penalty=args.shot_base_penalty,
        shot_miss_penalty_scale=args.shot_miss_penalty_scale,
        shot_miss_target_radius_mm=args.shot_miss_target_radius_mm,
        shot_miss_penalty_power=args.shot_miss_penalty_power,
        shot_miss_max_ratio=args.shot_miss_max_ratio,
        shot_miss_max_angle_deg=args.shot_miss_max_angle_deg,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        debug_interval=args.debug_interval,
        debug_window=args.debug_window,
        policy_std_decay=args.policy_std_decay,
        policy_std_min=args.policy_std_min,
        eval_patience=args.eval_patience,
        eval_min_delta=args.eval_min_delta,
        ppo_epochs=args.ppo_epochs,
        ppo_clip_coef=args.ppo_clip_coef,
        ppo_value_coef=args.ppo_value_coef,
        ppo_entropy_coef=args.ppo_entropy_coef,
        log_dir=args.log_dir,
    )
