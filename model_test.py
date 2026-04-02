import argparse
import numpy as np
import torch
from turret_env import TurretEnv
from turret_model_train import PolicyGRU


def infer_action_from_robot_inputs(
    model: PolicyGRU,
    yaw_to_panel: float,
    distance_to_panel_mm: float,
    panel_yaw_world: float,
    projectile_speed_mm_per_s: float = 23000.0,
    max_distance_mm: float = 1.0,
    obs_history: np.ndarray | None = None,
    obs_history_k: int = 4,
    device: torch.device | None = None,
    hidden: torch.Tensor | None = None,
):
    """
    Sample inference helper for a real robot loop.

    Args:
        model: Trained PolicyGRU.
        yaw_to_panel: Yaw from camera to panel center (radians, world frame).
        distance_to_panel_mm: Distance to panel center (mm).
        panel_yaw_world: Panel outward-normal yaw in world frame (radians).
        projectile_speed_mm_per_s: Projectile speed (mm/s), used for normalization.
        max_distance_mm: Max range for distance normalization (set to match training).
        obs_history: Optional (K, 4) history buffer (oldest -> newest).
        obs_history_k: Number of history steps (must match training, default 4).
        device: Torch device for inference.
        hidden: Optional GRU hidden state to carry between steps.

    Returns:
        action: np.array([yaw_correction, time_to_fire]) in model action space.
        hidden: Updated GRU hidden state.
        obs_history: Updated history buffer (K, 4).
    """
    if device is None:
        device = next(model.parameters()).device

    # Match training normalization.
    obs = np.array(
        [
            yaw_to_panel / np.pi,
            panel_yaw_world / np.pi,
            distance_to_panel_mm / max(float(max_distance_mm), 1e-6),
            projectile_speed_mm_per_s / 23000.0,
        ],
        dtype=np.float32,
    )

    # Maintain a short history window (oldest -> newest) to match training.
    k = int(obs_history_k)
    if k < 1:
        raise ValueError(f"obs_history_k must be >= 1, got {k}")
    if obs_history is None or obs_history.shape != (k, 4):
        obs_history = np.repeat(obs[None, :], k, axis=0)
    else:
        obs_history = np.roll(obs_history, shift=-1, axis=0)
        obs_history[-1] = obs

    obs_stacked = obs_history.reshape(-1).astype(np.float32, copy=False)
    obs_t = torch.as_tensor(obs_stacked, dtype=torch.float32, device=device).view(1, 1, -1)

    with torch.no_grad():
        action_t, hidden_new = model.mean_action(obs_t, hidden)
    action = action_t.squeeze(0).squeeze(0).cpu().numpy()
    return action, hidden_new, obs_history


def _remap_policy_keys(state: dict) -> dict:
    """Remap CUDA/actor-critic checkpoint keys to the eval policy keys.

    - mu_head.* -> mu.*
    - drop value_head.* (critic head)
    - drop _action_low/_action_high buffers
    """
    if not isinstance(state, dict):
        return state

    if "mu_head.weight" not in state and "mu.weight" in state:
        # Already in expected format.
        return state

    remapped: dict = {}
    for k, v in state.items():
        if k.startswith("value_head."):
            continue
        if k in {"_action_low", "_action_high"}:
            continue
        if k.startswith("mu_head."):
            remapped[k.replace("mu_head.", "mu.")] = v
            continue
        remapped[k] = v
    return remapped


def _load_policy_state_dict(model_path: str, device: torch.device, allow_unsafe: bool) -> dict:
    """Load either a raw state_dict or a training checkpoint with policy_state_dict.

    Uses weights_only=True when available for safety, and falls back to unsafe load
    only if explicitly allowed.
    """
    try:
        # PyTorch 2.6+ supports weights_only and safe globals context.
        from torch.serialization import safe_globals

        with safe_globals([np.core.multiarray.scalar, np.dtype, np.float64, np.float32]):
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch: weights_only not supported.
        ckpt = torch.load(model_path, map_location=device)
    except Exception:
        if not allow_unsafe:
            raise
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        state = ckpt["policy_state_dict"]
    else:
        state = ckpt

    return _remap_policy_keys(state)


def test_policy(
    model_path: str | None = None,
    n_episodes: int = 10,
    render: bool = True,
    hidden_dim: int = 128,
    allow_unsafe_load: bool = False,
):
    # Use the same obs_dim, action_dim and hidden_dim as in training
    action_dim = 2
    hidden_dim = int(hidden_dim)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Check your NVIDIA driver/CUDA setup.")
    device = torch.device("cuda")

    # Must match the action semantics used during training.
    env = TurretEnv(action_is_correction=True, correction_baseline="ideal")
    correction_yaw_max = float(getattr(env, "correction_clip_yaw", np.deg2rad(30.0)) or np.deg2rad(30.0))
    correction_pitch_max = float(getattr(env, "correction_clip_pitch", np.deg2rad(15.0)) or np.deg2rad(15.0))
    action_low = [-correction_yaw_max, 0.0]
    action_high = [correction_yaw_max, 1.0]

    obs_dim = int(getattr(env, "obs_dim", 4))
    if model_path is None:
        model_path = "best_policy_gru.pt"

    policy = PolicyGRU(
        obs_dim,
        hidden_dim,
        action_dim,
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    policy_state = _load_policy_state_dict(model_path, device, allow_unsafe_load)
    policy.load_state_dict(policy_state, strict=False)
    policy.eval()

    episode_returns = []
    episode_hit_pcts = []
    total_hits = 0
    total_shots = 0
    total_possible_hits = 0
    total_yaw_correction = 0.0
    total_steps = 0
    lead_angle_deg = 2.0  # match train_cuda default lead-reward-angle-scale
    lead_angle_rad = np.deg2rad(lead_angle_deg)
    for ep in range(n_episodes):
        obs = env.reset()
        # Per-episode hit/shots counters start at 0 after reset().
        start_hit_count = int(getattr(env.sim, "hit_count", 0) or 0)
        start_shots_fired = int(getattr(env.sim, "shots_fired", 0) or 0)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        hidden = None
        done = False
        ep_return = 0.0
        step = 0

        while not done:
            with torch.no_grad():
                action, hidden = policy.mean_action(obs, hidden)
            action_np = action.squeeze(0).squeeze(0).cpu().numpy()
            total_yaw_correction += float(action_np[0])
            total_steps += 1
            obs_next, reward, done, info = env.step(action_np)

            # Count "possible hits" when aligned with ideal lead
            ideal_yaw = getattr(env.sim, "ideal_yaw", None)
            cam = getattr(env.sim, "camera", None)
            if ideal_yaw is not None and cam is not None:
                yaw_err = np.arctan2(
                    np.sin(float(ideal_yaw) - float(cam.theta)),
                    np.cos(float(ideal_yaw) - float(cam.theta)),
                )
                if abs(yaw_err) <= lead_angle_rad:
                    total_possible_hits += 1

            ep_return += reward
            step += 1
            obs = torch.as_tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            if hasattr(env, "render") and render:
                env.render()
        end_hit_count = int(getattr(env.sim, "hit_count", 0) or 0)
        end_shots_fired = int(getattr(env.sim, "shots_fired", 0) or 0)
        ep_hits = end_hit_count - start_hit_count
        ep_shots = end_shots_fired - start_shots_fired
        total_hits += ep_hits
        total_shots += ep_shots
        hit_pct = 0.0 if ep_shots <= 0 else 100.0 * (ep_hits / float(ep_shots))
        episode_hit_pcts.append(hit_pct)
        print(
            f"Episode {ep + 1}: Return = {ep_return:.2f}, Steps = {step}, "
            f"Hits = {ep_hits}, Shots = {ep_shots}, Hit% = {hit_pct:.1f}%"
        )
        episode_returns.append(ep_return)
    print(f"Average Return over {n_episodes} episodes: {np.mean(episode_returns):.2f}")
    total_hit_pct = 0.0 if total_shots <= 0 else 100.0 * (total_hits / float(total_shots))
    print(f"Average Hit% over {n_episodes} episodes: {np.mean(episode_hit_pcts):.1f}%")
    print(f"Total Hit% over {n_episodes} episodes: {total_hit_pct:.1f}% ({total_hits}/{total_shots})")
    if total_possible_hits > 0:
        poss_eff = 100.0 * (total_hits / float(total_possible_hits))
        print(f"Total hits / possible hits: {total_hits}/{total_possible_hits} = {poss_eff:.1f}%")
    else:
        print("Total hits / possible hits: 0/0 (no aligned steps)")
    if total_steps > 0:
        mean_yaw_deg = np.rad2deg(total_yaw_correction / float(total_steps))
        print(f"Mean yaw correction (deg): {mean_yaw_deg:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow torch.load(weights_only=False) if safe load fails.",
    )
    args = parser.parse_args()

    test_policy(
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=not args.no_render,
        hidden_dim=args.hidden_dim,
        allow_unsafe_load=args.unsafe_load,
    )
