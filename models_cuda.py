"""
CUDA-Optimized Policy and Value Networks for Vectorized Training.

Features:
- Batched inference for N parallel environments
- Mixed precision (AMP) support
- Efficient hidden state management
- TorchScript-compatible operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import math


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

@torch.jit.script
def squash_to_range(
    raw_action: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Map unbounded action to [low, high] using tanh squashing."""
    action01 = 0.5 * (torch.tanh(raw_action) + 1.0)
    return low + (high - low) * action01


@torch.jit.script
def compute_log_prob_with_squash(
    raw_action: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability with tanh squashing correction.
    
    Args:
        raw_action: Pre-tanh action values
        mu: Distribution mean
        std: Distribution std
        
    Returns:
        log_prob: (batch,) log probabilities
    """
    # Gaussian log prob
    var = std * std
    log_scale = torch.log(std)
    log_prob = -0.5 * (((raw_action - mu) ** 2) / var + 2 * log_scale + math.log(2 * math.pi))
    log_prob = log_prob.sum(dim=-1)
    
    # Tanh squashing correction: log(1 - tanh(x)^2)
    # Numerically stable form
    log_det_jacobian = 2.0 * (
        math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action)
    )
    log_prob = log_prob - log_det_jacobian.sum(dim=-1)
    
    return log_prob


@torch.jit.script
def inverse_squash(
    action: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Invert tanh+affine squashing to recover raw action."""
    denom = torch.clamp(high - low, min=eps)
    action01 = torch.clamp((action - low) / denom, eps, 1.0 - eps)
    tanh_raw = 2.0 * action01 - 1.0
    tanh_raw = torch.clamp(tanh_raw, -1.0 + eps, 1.0 - eps)
    # atanh(x) = 0.5 * log((1+x)/(1-x))
    raw = 0.5 * (torch.log1p(tanh_raw) - torch.log1p(-tanh_raw))
    return raw


# ============================================================
# BASE POLICY CLASS
# ============================================================

class BasePolicyCUDA(nn.Module):
    """Base class for all policy networks with common functionality."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: Optional[list] = None,
        action_high: Optional[list] = None,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Default action bounds
        if action_low is None:
            action_low = [-math.radians(30), -math.radians(15), 0.0]
        if action_high is None:
            action_high = [math.radians(30), math.radians(15), 1.0]
        
        # Register as buffers so they move with the model
        self.register_buffer(
            '_action_low', 
            torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            '_action_high', 
            torch.tensor(action_high, dtype=torch.float32)
        )
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def _squash_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Squash raw action to action space."""
        return squash_to_range(raw_action, self._action_low, self._action_high)
    
    def _get_std(self) -> torch.Tensor:
        """Get current standard deviation."""
        return self.log_std.exp()
    
    def set_std(self, std: float) -> None:
        """Set standard deviation (for std decay during training)."""
        with torch.no_grad():
            self.log_std.fill_(math.log(std))
    
    def decay_std(self, decay_factor: float, min_std: float = 0.1) -> None:
        """Decay standard deviation by factor with minimum."""
        with torch.no_grad():
            current_std = self.log_std.exp()
            new_std = torch.clamp(current_std * decay_factor, min=min_std)
            self.log_std.copy_(new_std.log())


# ============================================================
# GRU POLICY
# ============================================================

class PolicyGRUCUDA(BasePolicyCUDA):
    """
    GRU-based policy network optimized for batched inference.
    
    Supports:
    - Batched inference over N environments
    - Sequence processing for trajectory recomputation
    - Mixed precision training
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 1,
        action_low: Optional[list] = None,
        action_high: Optional[list] = None,
    ):
        super().__init__(obs_dim, action_dim, action_low, action_high)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Network layers
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # Orthogonal init for RNN weights
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state for batch_size environments."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=device, dtype=next(self.parameters()).dtype
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: (N, T, obs_dim) or (N, obs_dim) observations
            hidden: (num_layers, N, hidden_dim) GRU hidden state
            
        Returns:
            mu: (N, T, action_dim) or (N, action_dim) mean actions
            std: (action_dim,) standard deviations
            hidden: (num_layers, N, hidden_dim) new hidden state
        """
        # Handle 2D input (N, obs_dim) -> (N, 1, obs_dim)
        squeeze_output = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze_output = True
        
        batch_size = obs.shape[0]
        
        # Initialize hidden if needed
        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)
        
        # Embed observations
        x = F.relu(self.obs_embed(obs))  # (N, T, hidden_dim)
        
        # GRU forward
        gru_out, hidden_new = self.gru(x, hidden)  # (N, T, hidden_dim)
        
        # Compute mean
        mu = self.mu_head(gru_out)  # (N, T, action_dim)
        
        # Get std
        std = self._get_std()
        
        if squeeze_output:
            mu = mu.squeeze(1)
        
        return mu, std, hidden_new
    
    def sample_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions for all environments.
        
        Args:
            obs: (N, obs_dim) or (N, T, obs_dim) observations
            hidden: (num_layers, N, hidden_dim) hidden state
            deterministic: If True, return mean action
            
        Returns:
            action: (N, action_dim) or (N, T, action_dim) squashed actions
            log_prob: (N,) or (N, T) log probabilities
            hidden: (num_layers, N, hidden_dim) new hidden state
        """
        mu, std, hidden_new = self.forward(obs, hidden)
        
        if deterministic:
            action = self._squash_action(mu)
            # Log prob not meaningful for deterministic
            log_prob = torch.zeros(mu.shape[:-1], device=mu.device)
        else:
            # Sample from Gaussian
            noise = torch.randn_like(mu)
            raw_action = mu + std * noise
            
            # Squash to action space
            action = self._squash_action(raw_action)
            
            # Compute log probability
            log_prob = compute_log_prob_with_squash(raw_action, mu, std)
        
        return action, log_prob, hidden_new
    
    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for action sequences.
        Used in PPO for recomputing log probs on stored trajectories.
        
        Args:
            obs_seq: (N, T, obs_dim) observation sequences
            actions_seq: (N, T, action_dim) squashed action sequences
            hidden: Initial hidden state
            
        Returns:
            log_probs: (N, T) log probabilities
            entropy: (N, T) entropy values
            hidden: Final hidden state
        """
        mu, std, hidden_new = self.forward(obs_seq, hidden)
        
        # Invert squashing to get raw actions
        raw_actions = inverse_squash(actions_seq, self._action_low, self._action_high)
        
        # Compute log probs
        log_probs = compute_log_prob_with_squash(raw_actions, mu, std)
        
        # Compute entropy (Gaussian entropy, not accounting for squash)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)
        entropy = entropy.sum() * torch.ones_like(log_probs)
        
        return log_probs, entropy, hidden_new
    
    def mean_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get deterministic mean action."""
        action, _, hidden_new = self.sample_action(obs, hidden, deterministic=True)
        return action, hidden_new


# ============================================================
# LSTM POLICY
# ============================================================

class PolicyLSTMCUDA(BasePolicyCUDA):
    """
    LSTM-based policy network optimized for batched inference.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 1,
        action_low: Optional[list] = None,
        action_high: Optional[list] = None,
    ):
        super().__init__(obs_dim, action_dim, action_low, action_high)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def init_hidden(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize (h, c) hidden state tuple."""
        dtype = next(self.parameters()).dtype
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        return (h, c)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        squeeze_output = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze_output = True
        
        batch_size = obs.shape[0]
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)
        
        x = F.relu(self.obs_embed(obs))
        lstm_out, hidden_new = self.lstm(x, hidden)
        mu = self.mu_head(lstm_out)
        std = self._get_std()
        
        if squeeze_output:
            mu = mu.squeeze(1)
        
        return mu, std, hidden_new
    
    def sample_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample actions."""
        mu, std, hidden_new = self.forward(obs, hidden)
        
        if deterministic:
            action = self._squash_action(mu)
            log_prob = torch.zeros(mu.shape[:-1], device=mu.device)
        else:
            noise = torch.randn_like(mu)
            raw_action = mu + std * noise
            action = self._squash_action(raw_action)
            log_prob = compute_log_prob_with_squash(raw_action, mu, std)
        
        return action, log_prob, hidden_new
    
    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Evaluate actions for PPO."""
        mu, std, hidden_new = self.forward(obs_seq, hidden)
        raw_actions = inverse_squash(actions_seq, self._action_low, self._action_high)
        log_probs = compute_log_prob_with_squash(raw_actions, mu, std)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum()
        entropy = entropy * torch.ones_like(log_probs)
        return log_probs, entropy, hidden_new
    
    def mean_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get deterministic mean action."""
        action, _, hidden_new = self.sample_action(obs, hidden, deterministic=True)
        return action, hidden_new


# ============================================================
# MLP POLICY (No Recurrence)
# ============================================================

class PolicyMLPCUDA(BasePolicyCUDA):
    """
    MLP-based policy (no recurrence).
    Can optionally use observation stacking.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        n_layers: int = 2,
        obs_history_k: int = 1,
        action_low: Optional[list] = None,
        action_high: Optional[list] = None,
    ):
        # If using history, multiply obs_dim
        effective_obs_dim = obs_dim * obs_history_k
        super().__init__(effective_obs_dim, action_dim, action_low, action_high)
        
        self.base_obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.obs_history_k = obs_history_k
        
        # Build MLP layers
        layers = []
        in_dim = effective_obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Forward pass.
        
        Args:
            obs: (N, obs_dim) or (N, T, obs_dim) observations
                 If using obs_history_k > 1, expects (N, obs_dim * k)
            hidden: Ignored (for API compatibility)
            
        Returns:
            mu: (N, action_dim) mean actions
            std: (action_dim,) standard deviations
            hidden: None (no recurrence)
        """
        # Handle sequence input - take last timestep
        if obs.dim() == 3:
            obs = obs[:, -1, :]
        
        x = self.mlp(obs)
        mu = self.mu_head(x)
        std = self._get_std()
        
        return mu, std, None
    
    def sample_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Sample actions."""
        mu, std, _ = self.forward(obs, hidden)
        
        if deterministic:
            action = self._squash_action(mu)
            log_prob = torch.zeros(mu.shape[0], device=mu.device)
        else:
            noise = torch.randn_like(mu)
            raw_action = mu + std * noise
            action = self._squash_action(raw_action)
            log_prob = compute_log_prob_with_squash(raw_action, mu, std)
        
        return action, log_prob, None
    
    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Evaluate actions for PPO."""
        # For MLP, process each timestep independently
        if obs_seq.dim() == 3:
            N, T, _ = obs_seq.shape
            obs_flat = obs_seq.reshape(N * T, -1)
            actions_flat = actions_seq.reshape(N * T, -1)
            
            mu, std, _ = self.forward(obs_flat)
            raw_actions = inverse_squash(actions_flat, self._action_low, self._action_high)
            log_probs = compute_log_prob_with_squash(raw_actions, mu, std)
            
            log_probs = log_probs.reshape(N, T)
            entropy = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum()
            entropy = entropy * torch.ones_like(log_probs)
        else:
            mu, std, _ = self.forward(obs_seq)
            raw_actions = inverse_squash(actions_seq, self._action_low, self._action_high)
            log_probs = compute_log_prob_with_squash(raw_actions, mu, std)
            entropy = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum()
            entropy = entropy * torch.ones_like(log_probs)
        
        return log_probs, entropy, None
    
    def mean_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Get deterministic mean action."""
        action, _, _ = self.sample_action(obs, hidden, deterministic=True)
        return action, None


# ============================================================
# VALUE NETWORKS
# ============================================================

class ValueMLPCUDA(nn.Module):
    """
    MLP-based value network for PPO.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        
        # Build MLP
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimates.
        
        Args:
            obs: (N,) or (N, T, obs_dim) or (N, obs_dim) observations
            
        Returns:
            values: (N,) or (N, T) value estimates
        """
        if obs.dim() == 3:
            # Sequence input: (N, T, obs_dim)
            N, T, D = obs.shape
            obs_flat = obs.reshape(N * T, D)
            values = self.value_head(self.mlp(obs_flat)).squeeze(-1)
            values = values.reshape(N, T)
        else:
            values = self.value_head(self.mlp(obs)).squeeze(-1)
        
        return values


class ValueGRUCUDA(nn.Module):
    """
    GRU-based value network for PPO with recurrent policies.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        dtype = next(self.parameters()).dtype
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute value estimates.
        
        Args:
            obs: (N, T, obs_dim) or (N, obs_dim) observations
            hidden: (num_layers, N, hidden_dim) hidden state
            
        Returns:
            values: (N, T) or (N,) value estimates
            hidden: New hidden state
        """
        squeeze_output = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze_output = True
        
        batch_size = obs.shape[0]
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)
        
        x = F.relu(self.obs_embed(obs))
        gru_out, hidden_new = self.gru(x, hidden)
        values = self.value_head(gru_out).squeeze(-1)
        
        if squeeze_output:
            values = values.squeeze(1)
        
        return values, hidden_new


# ============================================================
# COMBINED ACTOR-CRITIC
# ============================================================

class ActorCriticGRUCUDA(nn.Module):
    """
    Combined Actor-Critic with shared GRU backbone.
    More efficient than separate networks for PPO.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 1,
        action_low: Optional[list] = None,
        action_high: Optional[list] = None,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        
        # Default action bounds
        if action_low is None:
            action_low = [-math.radians(30), -math.radians(15), 0.0]
        if action_high is None:
            action_high = [math.radians(30), math.radians(15), 1.0]
        
        self.register_buffer('_action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('_action_high', torch.tensor(action_high, dtype=torch.float32))
        
        # Shared backbone
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Separate heads
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
    
    def _squash_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return squash_to_range(raw_action, self._action_low, self._action_high)
    
    def _get_std(self) -> torch.Tensor:
        return self.log_std.exp()
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy and value outputs.
        
        Returns:
            mu: Mean actions
            std: Std deviations
            values: Value estimates
            hidden: New hidden state
        """
        squeeze_output = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze_output = True
        
        batch_size = obs.shape[0]
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)
        
        # Shared forward pass
        x = F.relu(self.obs_embed(obs))
        gru_out, hidden_new = self.gru(x, hidden)
        
        # Separate heads
        mu = self.mu_head(gru_out)
        values = self.value_head(gru_out).squeeze(-1)
        std = self._get_std()
        
        if squeeze_output:
            mu = mu.squeeze(1)
            values = values.squeeze(1)
        
        return mu, std, values, hidden_new
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute value in single forward pass.
        
        Returns:
            action: Squashed actions
            log_prob: Log probabilities
            value: Value estimates
            hidden: New hidden state
        """
        mu, std, value, hidden_new = self.forward(obs, hidden)
        
        if deterministic:
            action = self._squash_action(mu)
            log_prob = torch.zeros(mu.shape[:-1], device=mu.device)
        else:
            noise = torch.randn_like(mu)
            raw_action = mu + std * noise
            action = self._squash_action(raw_action)
            log_prob = compute_log_prob_with_squash(raw_action, mu, std)
        
        return action, log_prob, value, hidden_new
    
    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions and compute values for PPO update.
        
        Returns:
            log_probs: Log probabilities
            entropy: Entropy values
            values: Value estimates
            hidden: Final hidden state
        """
        mu, std, values, hidden_new = self.forward(obs_seq, hidden)
        
        raw_actions = inverse_squash(actions_seq, self._action_low, self._action_high)
        log_probs = compute_log_prob_with_squash(raw_actions, mu, std)
        
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum()
        entropy = entropy * torch.ones_like(log_probs)
        
        return log_probs, entropy, values, hidden_new
    
    def decay_std(self, decay_factor: float, min_std: float = 0.1) -> None:
        """Decay standard deviation."""
        with torch.no_grad():
            current_std = self.log_std.exp()
            new_std = torch.clamp(current_std * decay_factor, min=min_std)
            self.log_std.copy_(new_std.log())


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_policy(
    model_type: str,
    obs_dim: int,
    hidden_dim: int,
    action_dim: int,
    device: torch.device,
    action_low: Optional[list] = None,
    action_high: Optional[list] = None,
    **kwargs,
) -> BasePolicyCUDA:
    """
    Factory function to create policy networks.
    
    Args:
        model_type: "gru", "lstm", or "mlp"
        obs_dim: Observation dimension
        hidden_dim: Hidden layer dimension
        action_dim: Action dimension
        device: PyTorch device
        action_low: Action space lower bounds
        action_high: Action space upper bounds
        **kwargs: Additional model-specific arguments
        
    Returns:
        Policy network on specified device
    """
    model_type = model_type.lower().strip()
    
    if model_type == "gru":
        policy = PolicyGRUCUDA(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            num_layers=kwargs.get("num_layers", 1),
        )
    elif model_type == "lstm":
        policy = PolicyLSTMCUDA(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            num_layers=kwargs.get("num_layers", 1),
        )
    elif model_type == "mlp":
        policy = PolicyMLPCUDA(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            n_layers=kwargs.get("n_layers", 2),
            obs_history_k=kwargs.get("obs_history_k", 1),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return policy.to(device)


def create_value_network(
    model_type: str,
    obs_dim: int,
    hidden_dim: int,
    device: torch.device,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create value networks.
    
    Args:
        model_type: "mlp" or "gru"
        obs_dim: Observation dimension
        hidden_dim: Hidden layer dimension
        device: PyTorch device
        
    Returns:
        Value network on specified device
    """
    model_type = model_type.lower().strip()
    
    if model_type == "mlp":
        value_net = ValueMLPCUDA(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            n_layers=kwargs.get("n_layers", 2),
        )
    elif model_type == "gru":
        value_net = ValueGRUCUDA(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=kwargs.get("num_layers", 1),
        )
    else:
        raise ValueError(f"Unknown model_type for value network: {model_type}")
    
    return value_net.to(device)


def create_actor_critic(
    obs_dim: int,
    hidden_dim: int,
    action_dim: int,
    device: torch.device,
    action_low: Optional[list] = None,
    action_high: Optional[list] = None,
    num_layers: int = 1,
) -> ActorCriticGRUCUDA:
    """Create combined actor-critic network."""
    return ActorCriticGRUCUDA(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        num_layers=num_layers,
        action_low=action_low,
        action_high=action_high,
    ).to(device)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    N_ENVS = 256
    OBS_DIM = 7
    HIDDEN_DIM = 256
    ACTION_DIM = 3
    SEQ_LEN = 50
    
    print("\n=== Testing PolicyGRUCUDA ===")
    policy_gru = create_policy("gru", OBS_DIM, HIDDEN_DIM, ACTION_DIM, device)
    print(f"Parameters: {sum(p.numel() for p in policy_gru.parameters()):,}")
    
    # Test single-step inference
    obs = torch.randn(N_ENVS, OBS_DIM, device=device)
    hidden = None
    
    action, log_prob, hidden = policy_gru.sample_action(obs, hidden)
    print(f"Single step - Action: {action.shape}, LogProb: {log_prob.shape}, Hidden: {hidden.shape}")
    
    # Test sequence inference
    obs_seq = torch.randn(N_ENVS, SEQ_LEN, OBS_DIM, device=device)
    action_seq, log_prob_seq, hidden = policy_gru.sample_action(obs_seq, None)
    print(f"Sequence - Action: {action_seq.shape}, LogProb: {log_prob_seq.shape}")
    
    # Test action evaluation (for PPO)
    log_probs, entropy, _ = policy_gru.evaluate_actions(obs_seq, action_seq, None)
    print(f"Evaluate - LogProbs: {log_probs.shape}, Entropy: {entropy.shape}")
    
    print("\n=== Testing PolicyLSTMCUDA ===")
    policy_lstm = create_policy("lstm", OBS_DIM, HIDDEN_DIM, ACTION_DIM, device)
    print(f"Parameters: {sum(p.numel() for p in policy_lstm.parameters()):,}")
    
    action, log_prob, hidden = policy_lstm.sample_action(obs, None)
    print(f"Single step - Action: {action.shape}, Hidden: (h={hidden[0].shape}, c={hidden[1].shape})")
    
    print("\n=== Testing PolicyMLPCUDA ===")
    policy_mlp = create_policy("mlp", OBS_DIM, HIDDEN_DIM, ACTION_DIM, device, obs_history_k=4)
    print(f"Parameters: {sum(p.numel() for p in policy_mlp.parameters()):,}")
    
    obs_stacked = torch.randn(N_ENVS, OBS_DIM * 4, device=device)
    action, log_prob, _ = policy_mlp.sample_action(obs_stacked)
    print(f"MLP - Action: {action.shape}, LogProb: {log_prob.shape}")
    
    print("\n=== Testing ValueMLPCUDA ===")
    value_mlp = create_value_network("mlp", OBS_DIM, HIDDEN_DIM, device)
    print(f"Parameters: {sum(p.numel() for p in value_mlp.parameters()):,}")
    
    values = value_mlp(obs)
    print(f"Values shape: {values.shape}")
    
    values_seq = value_mlp(obs_seq)
    print(f"Values seq shape: {values_seq.shape}")
    
    print("\n=== Testing ActorCriticGRUCUDA ===")
    ac = create_actor_critic(OBS_DIM, HIDDEN_DIM, ACTION_DIM, device)
    print(f"Parameters: {sum(p.numel() for p in ac.parameters()):,}")
    
    action, log_prob, value, hidden = ac.get_action_and_value(obs, None)
    print(f"AC single - Action: {action.shape}, Value: {value.shape}, Hidden: {hidden.shape}")
    
    # Benchmark
    print("\n=== Benchmark ===")
    import time
    
    N_ITERS = 1000
    
    # Warm-up
    for _ in range(100):
        action, log_prob, hidden = policy_gru.sample_action(obs, hidden)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    
    hidden = None
    for _ in range(N_ITERS):
        action, log_prob, hidden = policy_gru.sample_action(obs, hidden)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start
    
    total_samples = N_ENVS * N_ITERS
    samples_per_sec = total_samples / elapsed
    
    print(f"GRU Policy: {samples_per_sec:,.0f} samples/sec ({N_ENVS} envs x {N_ITERS} iters)")
    
    # Test with mixed precision
    print("\n=== Mixed Precision Test ===")
    from torch.cuda.amp import autocast
    
    with autocast(enabled=True):
        action_fp16, log_prob_fp16, hidden_fp16 = policy_gru.sample_action(obs, None)
        print(f"FP16 Action dtype: {action_fp16.dtype}")
    
    print("\n=== All tests passed! ===")