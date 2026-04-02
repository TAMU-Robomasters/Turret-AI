import numpy as np
import torch
import torch.nn as nn

# === Model loading config (edit for your robot setup) ===
MODEL_PATH = "checkpoints/best_policy.pt"
OBS_HISTORY_K = 4
HIDDEN_DIM = 128
NUM_LAYERS = 50
MAX_DISTANCE_MM = 12000
PROJECTILE_SPEED_MM_PER_S = 23000.0

# Cached model (loaded once per process)
_MODEL: "PolicyGRU | None" = None
_DEVICE: torch.device | None = None


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

def infer_action_from_robot_inputs(
    yaw_to_panel: float,
    distance_to_panel_mm: float,
    panel_yaw_world: float,
    obs_history: np.ndarray | None = None,
    hidden: torch.Tensor | None = None,
):
    """
    Sample inference helper for a real robot loop.

    Args:
        yaw_to_panel: Yaw from camera to panel center (radians, world frame).
        distance_to_panel_mm: Distance to panel center (mm).
        panel_yaw_world: Panel outward-normal yaw in world frame (radians).
        obs_history: Optional (K, 4) history buffer (oldest -> newest).
        hidden: Optional GRU hidden state to carry between steps.

    Returns:
        action: np.array([yaw_correction, time_to_fire]) in model action space.
        hidden: Updated GRU hidden state.
        obs_history: Updated history buffer (K, 4).
    """
    model = _get_or_load_model()
    device = next(model.parameters()).device

    # Match training normalization.
    obs = np.array(
        [
            yaw_to_panel / np.pi,
            panel_yaw_world / np.pi,
            distance_to_panel_mm / max(float(MAX_DISTANCE_MM), 1e-6),
            PROJECTILE_SPEED_MM_PER_S / 23000.0,
        ],
        dtype=np.float32,
    )

    # Maintain a short history window (oldest -> newest) to match training.
    k = int(OBS_HISTORY_K)
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


def load_policy_for_robot(
    model_path: str,
    obs_history_k: int = 4,
    hidden_dim: int = 128,
    num_layers: int = 1,
    device: torch.device | None = None,
):
    """
    Load a trained PolicyGRU for robot inference.

    Returns:
        model: PolicyGRU ready for inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = 4 * int(obs_history_k)
    action_dim = 2  # yaw correction + time_to_fire
    action_low = [-np.deg2rad(30.0), 0.0]
    action_high = [np.deg2rad(30.0), 1.0]

    model = PolicyGRU(
        obs_dim=obs_dim,
        hidden_dim=int(hidden_dim),
        action_dim=action_dim,
        num_layers=int(num_layers),
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    try:
        from torch.serialization import safe_globals
        with safe_globals([np.core.multiarray.scalar, np.dtype, np.float64, np.float32]):
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    state = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _get_or_load_model() -> PolicyGRU:
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = load_policy_for_robot(
            MODEL_PATH,
            obs_history_k=OBS_HISTORY_K,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device=_DEVICE,
        )
    return _MODEL
