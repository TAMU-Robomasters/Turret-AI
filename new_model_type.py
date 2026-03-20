from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyPINT(nn.Module):
    """
    PINT (Policy Inference with a Neural Transformer).

    This is a drop-in alternative to `PolicyGRU` that keeps the same public API:
      - sample_action(obs_seq, hidden=None) -> (action, log_prob, hidden)
      - mean_action(obs_seq, hidden=None)   -> (action, hidden)

    `hidden` is used as a rolling observation window (Tensor of shape BxTxObs),
    rather than an RNN hidden-state.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        context_len: int = 32,
        dropout: float = 0.1,
        action_low=None,
        action_high=None,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by n_heads; got hidden_dim={hidden_dim} n_heads={n_heads}"
            )
        if context_len <= 0:
            raise ValueError(f"context_len must be > 0, got {context_len}")

        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.context_len = int(context_len)

        self.obs_embed = nn.Linear(self.obs_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(self.context_len, self.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(n_heads),
            dim_feedforward=int(4 * self.hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Disable nested tensor path to avoid a warning when `norm_first=True`
        # (it is an optimization detail; correctness is unaffected).
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(n_layers),
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.mu = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        if action_low is None:
            action_low = [-np.pi, -np.pi / 3.0, 0.0]
        if action_high is None:
            action_high = [np.pi, np.pi / 3.0, 1.0]
        self._action_low_np = np.asarray(action_low, dtype=np.float32)
        self._action_high_np = np.asarray(action_high, dtype=np.float32)

        # Small init for positional embeddings.
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def _action_low(self, device):
        return torch.as_tensor(self._action_low_np, dtype=torch.float32, device=device)

    def _action_high(self, device):
        return torch.as_tensor(self._action_high_np, dtype=torch.float32, device=device)

    def _squash_to_action_space(self, raw_action: torch.Tensor) -> torch.Tensor:
        low = self._action_low(raw_action.device)
        high = self._action_high(raw_action.device)
        action01 = 0.5 * (torch.tanh(raw_action) + 1.0)
        return low + (high - low) * action01

    def _append_to_memory(
        self, obs_seq: torch.Tensor, memory: torch.Tensor | None
    ) -> torch.Tensor:
        if memory is None:
            full = obs_seq
        else:
            if not torch.is_tensor(memory):
                raise TypeError(
                    f"PolicyPINT hidden must be a Tensor (rolling obs window), got {type(memory)!r}"
                )
            if memory.dim() != 3:
                raise ValueError(
                    f"PolicyPINT hidden must have shape (B,T,Obs), got {tuple(memory.shape)}"
                )
            if memory.shape[0] != obs_seq.shape[0] or memory.shape[2] != obs_seq.shape[2]:
                raise ValueError(
                    "PolicyPINT hidden shape must match obs batch/obs_dim; "
                    f"hidden={tuple(memory.shape)} obs_seq={tuple(obs_seq.shape)}"
                )
            if memory.device != obs_seq.device:
                memory = memory.to(obs_seq.device)
            full = torch.cat([memory, obs_seq], dim=1)

        if full.shape[1] > self.context_len:
            full = full[:, -self.context_len :, :]
        return full.detach()

    def forward(self, obs_seq: torch.Tensor, hidden: torch.Tensor | None = None):
        if obs_seq.dim() != 3:
            raise ValueError(f"obs_seq must have shape (B,T,Obs), got {tuple(obs_seq.shape)}")

        ctx = self._append_to_memory(obs_seq, hidden)

        x = self.obs_embed(ctx)
        x = F.gelu(x)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:seq_len].unsqueeze(0)

        x = self.encoder(x)
        x = self.out_norm(x)
        x_last = x[:, -1, :]
        mu = self.mu(x_last).unsqueeze(1)
        return mu, self.log_std.exp(), ctx

    def sample_action(self, obs_seq: torch.Tensor, hidden: torch.Tensor | None = None):
        mu, std, new_hidden = self.forward(obs_seq, hidden)
        dist = torch.distributions.Normal(mu, std)

        # REINFORCE / score-function gradient: do not backprop through the sampled action.
        raw_action = dist.sample()
        action = self._squash_to_action_space(raw_action)

        # Tanh squashing correction for log-probability.
        log_det_jacobian = 2.0 * (
            np.log(2.0) - raw_action - torch.nn.functional.softplus(-2.0 * raw_action)
        )
        log_prob = dist.log_prob(raw_action).sum(-1) - log_det_jacobian.sum(-1)
        return action, log_prob, new_hidden

    def mean_action(self, obs_seq: torch.Tensor, hidden: torch.Tensor | None = None):
        mu, _, new_hidden = self.forward(obs_seq, hidden)
        return self._squash_to_action_space(mu), new_hidden
