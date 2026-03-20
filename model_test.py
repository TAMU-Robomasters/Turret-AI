import torch
import numpy as np
from turret_env import TurretEnv
from turret_model_train import PolicyGRU, PolicyLSTM, PolicyMLP
from new_model_type import PolicyPINT
import argparse

def test_policy(
    model: str = "gru",
    model_path: str | None = None,
    n_episodes: int = 10,
    render: bool = True,
    hidden_dim: int = 256,
    mlp_k: int = 8,
):
    # Use the same obs_dim, action_dim and hidden_dim as in training
    obs_dim_base = 7
    action_dim = 3
    hidden_dim = int(hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Must match the action semantics used during training.
    env = TurretEnv(action_is_correction=True, correction_baseline="panel")
    correction_yaw_max = float(getattr(env, "correction_clip_yaw", np.deg2rad(30.0)) or np.deg2rad(30.0))
    correction_pitch_max = float(getattr(env, "correction_clip_pitch", np.deg2rad(15.0)) or np.deg2rad(15.0))
    action_low = [-correction_yaw_max, -correction_pitch_max, 0.0]
    action_high = [correction_yaw_max, correction_pitch_max, 1.0]

    model = str(model).lower().strip()
    mlp_k = int(mlp_k)
    obs_dim = obs_dim_base * mlp_k if model == "mlp" else obs_dim_base
    if model_path is None:
        if model == "gru":
            model_path = "best_policy_gru.pt"
        elif model == "lstm":
            model_path = "best_policy_lstm.pt"
        elif model == "mlp":
            model_path = "best_policy_mlp.pt"
        else:
            model_path = "best_policy_pint.pt"

    if model == "gru":
        policy = PolicyGRU(
            obs_dim,
            hidden_dim,
            action_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
    elif model == "lstm":
        policy = PolicyLSTM(
            obs_dim,
            hidden_dim,
            action_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
    elif model == "mlp":
        policy = PolicyMLP(
            obs_dim,
            hidden_dim,
            action_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
    elif model == "pint":
        policy = PolicyPINT(
            obs_dim,
            hidden_dim,
            action_dim,
            n_layers=2,
            n_heads=4,
            context_len=32,
            dropout=0.1,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
    else:
        raise ValueError("Unknown model %r (expected 'gru', 'lstm', 'mlp', or 'pint')." % (model,))

    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    episode_returns = []
    episode_hit_pcts = []
    for ep in range(n_episodes):
        obs = env.reset()
        # Per-episode hit/shots counters start at 0 after reset().
        start_hit_count = int(getattr(env.sim, "hit_count", 0) or 0)
        start_shots_fired = int(getattr(env.sim, "shots_fired", 0) or 0)
        if model == "mlp" and mlp_k > 1:
            obs_hist = [np.asarray(obs, dtype=np.float32).copy() for _ in range(mlp_k)]
            obs_stack = np.concatenate(obs_hist, axis=0)
            obs = torch.as_tensor(obs_stack, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        hidden = None
        done = False
        ep_return = 0.0
        step = 0

        while not done:
            with torch.no_grad():
                action, hidden = policy.mean_action(obs, hidden)
            action_np = action.squeeze(0).squeeze(0).cpu().numpy()
            obs_next, reward, done, info = env.step(action_np)
            ep_return += reward
            step += 1
            if model == "mlp" and mlp_k > 1:
                obs_hist.pop(0)
                obs_hist.append(np.asarray(obs_next, dtype=np.float32))
                obs_stack_next = np.concatenate(obs_hist, axis=0)
                obs = torch.as_tensor(obs_stack_next, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            else:
                obs = torch.as_tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            if hasattr(env, "render") and render:
                env.render()
        end_hit_count = int(getattr(env.sim, "hit_count", 0) or 0)
        end_shots_fired = int(getattr(env.sim, "shots_fired", 0) or 0)
        ep_hits = end_hit_count - start_hit_count
        ep_shots = end_shots_fired - start_shots_fired
        hit_pct = 0.0 if ep_shots <= 0 else 100.0 * (ep_hits / float(ep_shots))
        episode_hit_pcts.append(hit_pct)
        print(
            f"Episode {ep + 1}: Return = {ep_return:.2f}, Steps = {step}, "
            f"Hits = {ep_hits}, Shots = {ep_shots}, Hit% = {hit_pct:.1f}%"
        )
        episode_returns.append(ep_return)
    print(f"Average Return over {n_episodes} episodes: {np.mean(episode_returns):.2f}")
    print(f"Average Hit% over {n_episodes} episodes: {np.mean(episode_hit_pcts):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm", "mlp", "pint"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-k", type=int, default=8)
    args = parser.parse_args()

    test_policy(
        model=args.model,
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=not args.no_render,
        hidden_dim=args.hidden_dim,
        mlp_k=args.mlp_k,
    )
