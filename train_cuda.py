"""
CUDA-Parallelized Training Loop for Vectorized Environments.

Features:
- Batched rollout collection from N parallel environments
- GPU-native PPO and REINFORCE implementations
- Mixed precision training (AMP)
- Efficient trajectory storage and processing
- Logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import argparse
from typing import Optional, Dict, Tuple, List
from collections import deque
import os

from vectorized_env_cuda import VectorizedTurretEnv, create_vectorized_env
from models_cuda import (
    create_policy,
    create_value_network,
    create_actor_critic,
    ActorCriticGRUCUDA,
)


# ============================================================
# ROLLOUT BUFFER
# ============================================================

class RolloutBuffer:
    """
    GPU-native rollout buffer for storing trajectories from vectorized environments.
    
    Stores complete rollouts from N environments for T steps.
    All data stays on GPU to minimize transfers.
    """
    
    def __init__(
        self,
        n_envs: int,
        n_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.dtype = dtype
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Pre-allocate tensors
        self.observations = torch.zeros(n_steps, n_envs, obs_dim, device=device, dtype=dtype)
        self.actions = torch.zeros(n_steps, n_envs, action_dim, device=device, dtype=dtype)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device, dtype=dtype)
        self.rewards = torch.zeros(n_steps, n_envs, device=device, dtype=dtype)
        self.dones = torch.zeros(n_steps, n_envs, device=device, dtype=torch.bool)
        self.values = torch.zeros(n_steps, n_envs, device=device, dtype=dtype)
        
        # Computed after rollout
        self.advantages = torch.zeros(n_steps, n_envs, device=device, dtype=dtype)
        self.returns = torch.zeros(n_steps, n_envs, device=device, dtype=dtype)
        
        # For recurrent policies - store hidden states
        self.hidden_states = None  # Will be set if using recurrent policy
        
        # Pointer
        self.ptr = 0
        self.full = False
    
    def reset(self):
        """Reset buffer pointer."""
        self.ptr = 0
        self.full = False
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Add a single timestep of data from all environments.
        
        Args:
            obs: (N, obs_dim)
            action: (N, action_dim)
            log_prob: (N,)
            reward: (N,)
            done: (N,)
            value: (N,)
        """
        if self.ptr >= self.n_steps:
            raise RuntimeError("Buffer overflow - call reset() or increase n_steps")
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
    ):
        """
        Compute returns and GAE advantages.
        
        Args:
            last_values: (N,) value estimates for the state after last step
            last_dones: (N,) done flags for the state after last step
        """
        # GAE computation
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.n_envs, device=self.device, dtype=self.dtype)
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = (~last_dones).float()
                next_values = last_values
            else:
                next_non_terminal = (~self.dones[t + 1]).float()
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        self.advantages = advantages
        self.returns = advantages + self.values
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        Generate mini-batches from the buffer.
        
        Yields:
            Dict with batch tensors
        """
        total_samples = self.n_steps * self.n_envs
        indices = torch.randperm(total_samples, device=self.device) if shuffle else torch.arange(total_samples, device=self.device)
        
        # Flatten time and env dimensions
        obs_flat = self.observations.reshape(total_samples, self.obs_dim)
        actions_flat = self.actions.reshape(total_samples, self.action_dim)
        log_probs_flat = self.log_probs.reshape(total_samples)
        advantages_flat = self.advantages.reshape(total_samples)
        returns_flat = self.returns.reshape(total_samples)
        values_flat = self.values.reshape(total_samples)
        
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            yield {
                "obs": obs_flat[batch_indices],
                "actions": actions_flat[batch_indices],
                "old_log_probs": log_probs_flat[batch_indices],
                "advantages": advantages_flat[batch_indices],
                "returns": returns_flat[batch_indices],
                "old_values": values_flat[batch_indices],
            }
    
    def get_sequences(
        self,
        seq_len: int,
    ):
        """
        Get data organized as sequences for recurrent policy training.
        
        Returns:
            Dict with sequence tensors of shape (n_seqs, seq_len, ...)
        """
        # Reshape to (n_steps, n_envs, ...) -> sequences
        # For simplicity, treat each environment's trajectory as a sequence
        # Pad/truncate to seq_len
        
        n_seqs_per_env = (self.n_steps + seq_len - 1) // seq_len
        
        # Pad if necessary
        pad_len = n_seqs_per_env * seq_len - self.n_steps
        
        if pad_len > 0:
            obs_padded = torch.cat([
                self.observations,
                torch.zeros(pad_len, self.n_envs, self.obs_dim, device=self.device, dtype=self.dtype)
            ], dim=0)
            actions_padded = torch.cat([
                self.actions,
                torch.zeros(pad_len, self.n_envs, self.action_dim, device=self.device, dtype=self.dtype)
            ], dim=0)
            log_probs_padded = torch.cat([
                self.log_probs,
                torch.zeros(pad_len, self.n_envs, device=self.device, dtype=self.dtype)
            ], dim=0)
            advantages_padded = torch.cat([
                self.advantages,
                torch.zeros(pad_len, self.n_envs, device=self.device, dtype=self.dtype)
            ], dim=0)
            returns_padded = torch.cat([
                self.returns,
                torch.zeros(pad_len, self.n_envs, device=self.device, dtype=self.dtype)
            ], dim=0)
            mask = torch.cat([
                torch.ones(self.n_steps, self.n_envs, device=self.device, dtype=torch.bool),
                torch.zeros(pad_len, self.n_envs, device=self.device, dtype=torch.bool)
            ], dim=0)
        else:
            obs_padded = self.observations
            actions_padded = self.actions
            log_probs_padded = self.log_probs
            advantages_padded = self.advantages
            returns_padded = self.returns
            mask = torch.ones(self.n_steps, self.n_envs, device=self.device, dtype=torch.bool)
        
        # Reshape to sequences: (n_seqs_per_env, seq_len, n_envs, ...)
        # Then transpose to (n_envs, n_seqs_per_env, seq_len, ...)
        # Finally flatten first two dims: (n_envs * n_seqs_per_env, seq_len, ...)
        
        total_len = n_seqs_per_env * seq_len
        
        obs_seqs = obs_padded[:total_len].reshape(n_seqs_per_env, seq_len, self.n_envs, self.obs_dim)
        obs_seqs = obs_seqs.permute(2, 0, 1, 3).reshape(-1, seq_len, self.obs_dim)
        
        actions_seqs = actions_padded[:total_len].reshape(n_seqs_per_env, seq_len, self.n_envs, self.action_dim)
        actions_seqs = actions_seqs.permute(2, 0, 1, 3).reshape(-1, seq_len, self.action_dim)
        
        log_probs_seqs = log_probs_padded[:total_len].reshape(n_seqs_per_env, seq_len, self.n_envs)
        log_probs_seqs = log_probs_seqs.permute(2, 0, 1).reshape(-1, seq_len)
        
        advantages_seqs = advantages_padded[:total_len].reshape(n_seqs_per_env, seq_len, self.n_envs)
        advantages_seqs = advantages_seqs.permute(2, 0, 1).reshape(-1, seq_len)
        
        returns_seqs = returns_padded[:total_len].reshape(n_seqs_per_env, seq_len, self.n_envs)
        returns_seqs = returns_seqs.permute(2, 0, 1).reshape(-1, seq_len)
        
        mask_seqs = mask[:total_len].reshape(n_seqs_per_env, seq_len, self.n_envs)
        mask_seqs = mask_seqs.permute(2, 0, 1).reshape(-1, seq_len)
        
        return {
            "obs": obs_seqs,
            "actions": actions_seqs,
            "old_log_probs": log_probs_seqs,
            "advantages": advantages_seqs,
            "returns": returns_seqs,
            "mask": mask_seqs,
        }


# ============================================================
# ROLLOUT COLLECTION
# ============================================================

@torch.no_grad()
def collect_rollouts(
    env: VectorizedTurretEnv,
    policy: nn.Module,
    buffer: RolloutBuffer,
    value_net: Optional[nn.Module] = None,
    n_steps: Optional[int] = None,
    deterministic: bool = False,
) -> Dict[str, float]:
    """
    Collect rollouts from vectorized environment.
    
    Args:
        env: Vectorized environment
        policy: Policy network
        buffer: Rollout buffer to fill
        value_net: Value network (if separate from policy)
        n_steps: Number of steps to collect (default: buffer.n_steps)
        deterministic: Use deterministic actions
        
    Returns:
        Dict with rollout statistics
    """
    if n_steps is None:
        n_steps = buffer.n_steps
    
    buffer.reset()
    
    # Get initial observation
    obs = env._get_obs()  # (N, obs_dim)
    
    # Initialize hidden state for recurrent policies
    hidden = None
    value_hidden = None
    
    # Check if policy is recurrent
    is_recurrent = hasattr(policy, 'init_hidden')
    is_actor_critic = isinstance(policy, ActorCriticGRUCUDA)
    
    # Stats tracking
    episode_rewards = []
    episode_lengths = []
    
    for step in range(n_steps):
        # Get action and value
        if is_actor_critic:
            action, log_prob, value, hidden = policy.get_action_and_value(
                obs, hidden, deterministic=deterministic
            )
        else:
            action, log_prob, hidden = policy.sample_action(obs, hidden, deterministic=deterministic)
            
            if value_net is not None:
                if hasattr(value_net, 'init_hidden'):
                    value, value_hidden = value_net(obs, value_hidden)
                else:
                    value = value_net(obs)
            else:
                value = torch.zeros(env.n_envs, device=env.device)
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        # Store in buffer
        buffer.add(
            obs=obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        )
        
        # Track completed episodes
        if "episode_done" in info:
            ep_done_mask = info["episode_done"]
            if ep_done_mask.any():
                ep_returns = info["episode_return"][ep_done_mask]
                ep_lengths = info["episode_length"][ep_done_mask]
                episode_rewards.extend(ep_returns.cpu().tolist())
                episode_lengths.extend(ep_lengths.cpu().tolist())
        
        # Reset hidden states for done environments (if recurrent)
        if is_recurrent and done.any():
            # Zero out hidden states for done envs
            done_idx = done.nonzero(as_tuple=True)[0]
            if hidden is not None:
                hidden[:, done_idx, :] = 0
        
        obs = next_obs
    
    # Compute last values for GAE
    with torch.no_grad():
        if is_actor_critic:
            _, _, last_value, _ = policy.get_action_and_value(obs, hidden, deterministic=True)
        elif value_net is not None:
            if hasattr(value_net, 'init_hidden'):
                last_value, _ = value_net(obs, value_hidden)
            else:
                last_value = value_net(obs)
        else:
            last_value = torch.zeros(env.n_envs, device=env.device)
    
    # Get last done flags
    # Note: These are the dones from the final step, not additional steps
    last_dones = done
    
    # Compute returns and advantages
    buffer.compute_returns_and_advantages(last_value, last_dones)
    
    # Compute statistics
    stats = {
        "rollout/reward_mean": buffer.rewards.mean().item(),
        "rollout/reward_std": buffer.rewards.std().item(),
        "rollout/value_mean": buffer.values.mean().item(),
        "rollout/advantage_mean": buffer.advantages.mean().item(),
        "rollout/return_mean": buffer.returns.mean().item(),
    }
    
    if episode_rewards:
        stats["rollout/ep_reward_mean"] = np.mean(episode_rewards)
        stats["rollout/ep_reward_std"] = np.std(episode_rewards) if len(episode_rewards) > 1 else 0
        stats["rollout/ep_length_mean"] = np.mean(episode_lengths)
        stats["rollout/episodes_completed"] = len(episode_rewards)
    
    return stats


# ============================================================
# PPO UPDATE
# ============================================================

def ppo_update(
    policy: nn.Module,
    value_net: Optional[nn.Module],
    optimizer: optim.Optimizer,
    value_optimizer: Optional[optim.Optimizer],
    buffer: RolloutBuffer,
    n_epochs: int = 4,
    batch_size: int = 256,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = None,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    Perform PPO update on collected rollouts.
    
    Args:
        policy: Policy network
        value_net: Value network (None if using ActorCritic)
        optimizer: Policy optimizer
        value_optimizer: Value optimizer (None if using ActorCritic)
        buffer: Filled rollout buffer
        n_epochs: Number of PPO epochs
        batch_size: Mini-batch size
        clip_coef: PPO clipping coefficient
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Early stopping KL threshold (None to disable)
        use_amp: Use automatic mixed precision
        scaler: GradScaler for AMP
        
    Returns:
        Dict with training statistics
    """
    is_actor_critic = isinstance(policy, ActorCriticGRUCUDA)
    
    # Track statistics
    policy_losses = []
    value_losses = []
    entropy_losses = []
    clip_fractions = []
    approx_kls = []
    
    for epoch in range(n_epochs):
        for batch in buffer.get_batches(batch_size, shuffle=True):
            obs = batch["obs"]
            actions = batch["actions"]
            old_log_probs = batch["old_log_probs"]
            advantages = batch["advantages"]
            returns = batch["returns"]
            old_values = batch["old_values"]
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            with autocast(enabled=use_amp):
                if is_actor_critic:
                    # Combined forward pass
                    log_probs, entropy, values, _ = policy.evaluate_actions(
                        obs.unsqueeze(1), actions.unsqueeze(1), None
                    )
                    log_probs = log_probs.squeeze(1)
                    entropy = entropy.squeeze(1)
                    values = values.squeeze(1)
                else:
                    # Separate policy and value
                    log_probs, entropy, _ = policy.evaluate_actions(
                        obs.unsqueeze(1), actions.unsqueeze(1), None
                    )
                    log_probs = log_probs.squeeze(1)
                    entropy = entropy.squeeze(1)
                    
                    if value_net is not None:
                        values = value_net(obs)
                    else:
                        values = torch.zeros_like(returns)
                
                # Policy loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (with optional clipping)
                value_loss = 0.5 * ((values - returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                if is_actor_critic or value_net is None:
                    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                else:
                    total_loss = policy_loss + entropy_coef * entropy_loss
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            if value_optimizer is not None:
                value_optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                scaler.scale(total_loss).backward()
                
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    if value_net is not None:
                        scaler.unscale_(value_optimizer)
                        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                if value_optimizer is not None and not is_actor_critic:
                    scaler.step(value_optimizer)
                scaler.update()
            else:
                total_loss.backward()
                
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    if value_net is not None and not is_actor_critic:
                        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                
                optimizer.step()
                if value_optimizer is not None and not is_actor_critic:
                    # Separate value update
                    if not is_actor_critic:
                        value_optimizer.zero_grad(set_to_none=True)
                        value_loss_only = 0.5 * ((values.detach() - returns) ** 2).mean()
                        # Recompute with fresh values
                        values_new = value_net(obs)
                        value_loss_new = 0.5 * ((values_new - returns) ** 2).mean()
                        value_loss_new.backward()
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                        value_optimizer.step()
            
            # Statistics
            with torch.no_grad():
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())
                
                # Clip fraction
                clip_fraction = ((ratio - 1).abs() > clip_coef).float().mean().item()
                clip_fractions.append(clip_fraction)
                
                # Approximate KL
                approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean().item()
                approx_kls.append(approx_kl)
        
        # Early stopping based on KL divergence
        if target_kl is not None and np.mean(approx_kls[-len(list(buffer.get_batches(batch_size))):]) > target_kl:
            print(f"Early stopping at epoch {epoch + 1} due to KL divergence")
            break
    
    return {
        "train/policy_loss": np.mean(policy_losses),
        "train/value_loss": np.mean(value_losses),
        "train/entropy": np.mean(entropy_losses),
        "train/clip_fraction": np.mean(clip_fractions),
        "train/approx_kl": np.mean(approx_kls),
        "train/n_epochs": epoch + 1,
    }


def ppo_update_recurrent(
    policy: nn.Module,
    value_net: Optional[nn.Module],
    optimizer: optim.Optimizer,
    value_optimizer: Optional[optim.Optimizer],
    buffer: RolloutBuffer,
    seq_len: int = 16,
    n_epochs: int = 4,
    batch_size: int = 32,  # Number of sequences per batch
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    PPO update for recurrent policies processing sequences.
    """
    is_actor_critic = isinstance(policy, ActorCriticGRUCUDA)
    
    # Get sequence data
    seq_data = buffer.get_sequences(seq_len)
    n_seqs = seq_data["obs"].shape[0]
    
    policy_losses = []
    value_losses = []
    entropy_losses = []
    clip_fractions = []
    
    for epoch in range(n_epochs):
        # Shuffle sequences
        perm = torch.randperm(n_seqs, device=buffer.device)
        
        for start in range(0, n_seqs, batch_size):
            end = min(start + batch_size, n_seqs)
            batch_idx = perm[start:end]
            
            obs_batch = seq_data["obs"][batch_idx]  # (B, T, obs_dim)
            actions_batch = seq_data["actions"][batch_idx]  # (B, T, action_dim)
            old_log_probs_batch = seq_data["old_log_probs"][batch_idx]  # (B, T)
            advantages_batch = seq_data["advantages"][batch_idx]  # (B, T)
            returns_batch = seq_data["returns"][batch_idx]  # (B, T)
            mask_batch = seq_data["mask"][batch_idx]  # (B, T)
            
            # Normalize advantages (only over valid entries)
            valid_adv = advantages_batch[mask_batch]
            if valid_adv.numel() > 1:
                adv_mean = valid_adv.mean()
                adv_std = valid_adv.std() + 1e-8
                advantages_batch = (advantages_batch - adv_mean) / adv_std
            
            with autocast(enabled=use_amp):
                if is_actor_critic:
                    log_probs, entropy, values, _ = policy.evaluate_actions(
                        obs_batch, actions_batch, None
                    )
                else:
                    log_probs, entropy, _ = policy.evaluate_actions(
                        obs_batch, actions_batch, None
                    )
                    if value_net is not None:
                        if hasattr(value_net, 'forward') and hasattr(value_net, 'init_hidden'):
                            values, _ = value_net(obs_batch, None)
                        else:
                            # Flatten, compute, reshape
                            B, T, D = obs_batch.shape
                            values = value_net(obs_batch.reshape(B * T, D)).reshape(B, T)
                    else:
                        values = torch.zeros_like(returns_batch)
                
                # Masked losses
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages_batch
                
                policy_loss_unmasked = -torch.min(surr1, surr2)
                policy_loss = (policy_loss_unmasked * mask_batch).sum() / mask_batch.sum()
                
                value_loss_unmasked = 0.5 * (values - returns_batch) ** 2
                value_loss = (value_loss_unmasked * mask_batch).sum() / mask_batch.sum()
                
                entropy_loss = -(entropy * mask_batch).sum() / mask_batch.sum()
                
                total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
            optimizer.zero_grad(set_to_none=True)
            if value_optimizer is not None:
                value_optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                scaler.step(optimizer)
                if value_optimizer is not None and not is_actor_critic:
                    scaler.unscale_(value_optimizer)
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                    scaler.step(value_optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                if value_optimizer is not None and not is_actor_critic:
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                    value_optimizer.step()
            
            with torch.no_grad():
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())
                clip_fraction = ((ratio - 1).abs() > clip_coef).float().mean().item()
                clip_fractions.append(clip_fraction)
    
    return {
        "train/policy_loss": np.mean(policy_losses),
        "train/value_loss": np.mean(value_losses),
        "train/entropy": np.mean(entropy_losses),
        "train/clip_fraction": np.mean(clip_fractions),
    }


# ============================================================
# REINFORCE UPDATE
# ============================================================

def reinforce_update(
    policy: nn.Module,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    REINFORCE (vanilla policy gradient) update.
    """
    # Flatten data
    total_samples = buffer.n_steps * buffer.n_envs
    
    log_probs = buffer.log_probs.reshape(total_samples)
    returns = buffer.returns.reshape(total_samples)
    
    # Normalize returns
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    with autocast(enabled=use_amp):
        # Policy gradient loss: -E[log_prob * return]
        loss = -(log_probs * returns_norm).mean()
    
    optimizer.zero_grad(set_to_none=True)
    
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
    
    return {
        "train/policy_loss": loss.item(),
        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "train/return_mean": returns.mean().item(),
        "train/return_std": returns.std().item(),
    }

# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_policy(
    env: VectorizedTurretEnv,
    policy: nn.Module,
    n_episodes: int = 100,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate policy over multiple episodes.
    
    Args:
        env: Vectorized environment
        policy: Policy network
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions
        
    Returns:
        Dict with evaluation statistics
    """
    policy.eval()
    
    # Reset all environments
    obs = env.reset()
    
    # Track episode stats
    episode_rewards = []
    episode_lengths = []
    episode_hits = []
    
    # Hidden state for recurrent policies
    hidden = None
    is_recurrent = hasattr(policy, 'init_hidden')
    
    # Run until we have enough episodes
    max_steps = env.max_steps * (n_episodes // env.n_envs + 2)
    
    for step in range(max_steps):
        if len(episode_rewards) >= n_episodes:
            break
        
        # Get action
        if hasattr(policy, 'get_action_and_value'):
            action, _, _, hidden = policy.get_action_and_value(obs, hidden, deterministic=deterministic)
        else:
            action, _, hidden = policy.sample_action(obs, hidden, deterministic=deterministic)
        
        # Step
        obs, reward, done, info = env.step(action)
        
        # Collect completed episodes
        if "episode_done" in info:
            ep_done = info["episode_done"]
            if ep_done.any():
                episode_rewards.extend(info["episode_return"][ep_done].cpu().tolist())
                episode_lengths.extend(info["episode_length"][ep_done].cpu().tolist())
                if "episode_hits" in info:
                    episode_hits.extend(info["episode_hits"][ep_done].cpu().tolist())
        
        # Reset hidden for done environments
        if is_recurrent and done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            if isinstance(hidden, tuple):
                hidden[0][:, done_idx, :] = 0
                hidden[1][:, done_idx, :] = 0
            elif hidden is not None:
                hidden[:, done_idx, :] = 0
    
    policy.train()
    
    # Compute statistics
    episode_rewards = episode_rewards[:n_episodes]
    episode_lengths = episode_lengths[:n_episodes]
    episode_hits = episode_hits[:n_episodes] if episode_hits else [0] * len(episode_rewards)
    
    return {
        "eval/reward_mean": np.mean(episode_rewards) if episode_rewards else 0,
        "eval/reward_std": np.std(episode_rewards) if len(episode_rewards) > 1 else 0,
        "eval/length_mean": np.mean(episode_lengths) if episode_lengths else 0,
        "eval/hits_mean": np.mean(episode_hits) if episode_hits else 0,
        "eval/n_episodes": len(episode_rewards),
    }


# ============================================================
# LOGGING UTILITIES
# ============================================================

class TrainingLogger:
    """Simple training logger with optional file output."""
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        print_interval: int = 1,
        append: bool = False,
    ):
        self.log_dir = log_dir
        self.print_interval = print_interval
        self.metrics_history = {}
        self.iteration = 0
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            mode = "a" if append else "w"
            self.log_file = open(os.path.join(log_dir, "training.log"), mode)
        else:
            self.log_file = None
    
    def log(self, metrics: Dict[str, float], iteration: Optional[int] = None):
        """Log metrics."""
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
        
        # Store history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((self.iteration, value))
        
        # Print
        if self.iteration % self.print_interval == 0:
            msg = f"[Iter {self.iteration}] " + " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            print(msg)
            
            if self.log_file is not None:
                self.log_file.write(msg + "\n")
                self.log_file.flush()
    
    def close(self):
        if self.log_file is not None:
            self.log_file.close()


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train(
    # Environment
    n_envs: int = 256,
    max_steps_per_episode: int = 500,
    dt: float = 0.015,
    # Training
    total_timesteps: int = 10_000_000,
    n_steps_per_rollout: int = 128,
    batch_size: int = 512,
    n_epochs: int = 2,
    learning_rate: float = 1e-4,
    # Adaptive LR (reward-plateau based, optional)
    adaptive_lr: bool = True,
    adaptive_lr_factor: float = 1.01,
    adaptive_lr_patience: int = 10,
    adaptive_lr_min_delta: float = 1.0,
    adaptive_lr_max: float = 1e-2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    # PPO
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = None,
    # Policy
    hidden_dim: int = 16,
    num_layers: int = 16,
    use_actor_critic: bool = True,
    # Exploration
    policy_std_init: float = 0.5,
    policy_std_decay: float = 0.9999,
    policy_std_min: float = 0.1,
    # Evaluation
    eval_interval: int = 10,
    eval_episodes: int = 50,
    eval_deterministic: bool = False,
    # Checkpointing
    save_interval: int = 50,
    save_dir: str = "checkpoints",
    resume: Optional[str] = None,
    # Misc
    seed: Optional[int] = None,
    use_amp: bool = True,
    log_interval: int = 1,
    # Reward shaping
    shot_base_penalty: float = 0.0,
    shot_miss_penalty_scale: float = 0.0,
    miss_penalty: float = 150.0,
    hit_reward: float = 200.0,
    predicted_yaw_slope: float = 10.0,
    predicted_yaw_tolerance_deg: float = 2.0,
):
    """
    Main training loop.
    
    Args:
        n_envs: Number of parallel environments
        total_timesteps: Total environment steps to train
        n_steps_per_rollout: Steps per rollout before update
        ... (see argument descriptions above)
    """
    # Setup device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Check your NVIDIA driver/CUDA setup.")
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Seeding
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Seed: {seed}")
    
    # Create environment
    print(f"\nCreating {n_envs} parallel environments...")
    env = create_vectorized_env(
        n_envs=n_envs,
        device=device,
        auto_reset=True,
        dt=dt,
        max_steps=max_steps_per_episode,
        obs_history_k=4,
        action_is_correction=True,
        shot_base_penalty=shot_base_penalty,
        shot_miss_penalty_scale=shot_miss_penalty_scale,
        miss_penalty=miss_penalty,
        hit_reward=hit_reward,
        predicted_yaw_slope=predicted_yaw_slope,
        predicted_yaw_tolerance_deg=predicted_yaw_tolerance_deg,
    )
    
    obs_dim = env.obs_dim
    action_dim = 2
    
    effective_obs_dim = obs_dim
    
    print(f"Observation dim: {obs_dim} (effective: {effective_obs_dim})")
    print(f"Action dim: {action_dim}")
    
    # Action bounds (corrections)
    correction_yaw_max = float(np.deg2rad(30.0))
    correction_pitch_max = float(np.deg2rad(15.0))
    action_low = [-correction_yaw_max, 0.0]
    action_high = [correction_yaw_max, 1.0]
    
    # Create networks
    print(f"\nCreating GRU policy (hidden_dim={hidden_dim})...")
    
    if use_actor_critic:
        # Combined actor-critic
        policy = create_actor_critic(
            obs_dim=effective_obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            device=device,
            action_low=action_low,
            action_high=action_high,
            num_layers=num_layers,
        )
        value_net = None
        
        # Single optimizer for combined network
        optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        value_optimizer = None
    else:
        # Separate policy and value networks
        policy = create_policy(
            model_type="gru",
            obs_dim=effective_obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            device=device,
            action_low=action_low,
            action_high=action_high,
            num_layers=num_layers,
            obs_history_k=4,
        )
        
        value_net = create_value_network(
            model_type="gru",
            obs_dim=effective_obs_dim,
            hidden_dim=hidden_dim,
            device=device,
            num_layers=num_layers,
        )
        
        optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    
    # Set initial std
    if hasattr(policy, 'log_std'):
        with torch.no_grad():
            policy.log_std.fill_(np.log(policy_std_init))
    
    # Print parameter counts
    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value_net.parameters()) if value_net else 0
    print(f"Policy parameters: {policy_params:,}")
    print(f"Value parameters: {value_params:,}")
    print(f"Total parameters: {policy_params + value_params:,}")
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        n_envs=n_envs,
        n_steps=n_steps_per_rollout,
        obs_dim=effective_obs_dim,
        action_dim=action_dim,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    
    # Mixed precision
    scaler = GradScaler() if use_amp and device.type == "cuda" else None
    print(f"Mixed precision (AMP): {use_amp and device.type == 'cuda'}")
    
    # Setup logging and checkpointing
    os.makedirs(save_dir, exist_ok=True)
    logger = TrainingLogger(log_dir=save_dir, print_interval=log_interval, append=bool(resume))
    
    # Training state
    best_eval_reward = -float('inf')
    last_eval_reward = None
    eval_plateau_count = 0
    total_steps = 0
    n_updates = 0
    start_time = time.time()
    start_iteration = 1
    
    # Calculate iterations needed
    steps_per_iteration = n_envs * n_steps_per_rollout
    total_iterations = total_timesteps // steps_per_iteration
    
    print(f"\n{'='*60}")
    print(f"Starting training:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Steps per iteration: {steps_per_iteration:,}")
    print(f"  Total iterations: {total_iterations:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  PPO epochs: {n_epochs}")
    print(f"{'='*60}\n")
    
    # Resume from checkpoint if provided
    if resume:
        ckpt = torch.load(resume, map_location=device)
        if "policy_state_dict" in ckpt:
            policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        if value_net is not None and ckpt.get("value_state_dict") is not None:
            value_net.load_state_dict(ckpt["value_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if value_optimizer is not None and ckpt.get("value_optimizer_state_dict") is not None:
            value_optimizer.load_state_dict(ckpt["value_optimizer_state_dict"])
        resume_iter = int(ckpt.get("iteration", 0) or 0)
        total_steps = int(ckpt.get("total_steps", resume_iter * steps_per_iteration) or 0)
        n_updates = int(ckpt.get("progress/updates", resume_iter) or resume_iter)
        best_eval_reward = float(ckpt.get("eval_reward", best_eval_reward))
        start_iteration = resume_iter + 1
        logger.iteration = resume_iter
        print(f"Resumed from iteration {resume_iter}, total_steps={total_steps:,}")
    
    # Reset environment
    env.reset()
    
    # Main training loop
    for iteration in range(start_iteration, total_iterations + 1):
        iter_start = time.time()
        
        # Collect rollouts
        policy.eval()
        
        rollout_stats = collect_rollouts(
            env=env,
            policy=policy,
            buffer=buffer,
            value_net=value_net,
        )
        
        total_steps += steps_per_iteration
        
        # Update policy
        policy.train()
        if value_net is not None:
            value_net.train()
        
        is_recurrent = True
        
        if is_recurrent and not isinstance(policy, ActorCriticGRUCUDA):
            # Use sequence-based PPO for recurrent
            train_stats = ppo_update_recurrent(
                policy=policy,
                value_net=value_net,
                optimizer=optimizer,
                value_optimizer=value_optimizer,
                buffer=buffer,
                seq_len=min(32, n_steps_per_rollout),
                n_epochs=n_epochs,
                batch_size=min(batch_size // 32, 64),
                clip_coef=clip_coef,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                use_amp=use_amp,
                scaler=scaler,
            )
        else:
            train_stats = ppo_update(
                policy=policy,
                value_net=value_net,
                optimizer=optimizer,
                value_optimizer=value_optimizer,
                buffer=buffer,
                n_epochs=n_epochs,
                batch_size=batch_size,
                clip_coef=clip_coef,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                use_amp=use_amp,
                scaler=scaler,
            )
        
        n_updates += 1
        
        # Decay exploration std
        if hasattr(policy, 'decay_std'):
            policy.decay_std(policy_std_decay, policy_std_min)

        
        # Compute timing
        iter_time = time.time() - iter_start
        total_time = time.time() - start_time
        steps_per_sec = total_steps / total_time
        # Get current std
        if hasattr(policy, '_get_std'):
            current_std = policy._get_std().mean().item()
        elif hasattr(policy, 'log_std'):
            current_std = policy.log_std.exp().mean().item()
        else:
            current_std = 0.0
        
        # Logging
        metrics = {
            **rollout_stats,
            **train_stats,
            "time/iteration": iter_time,
            "time/total": total_time,
            "time/steps_per_sec": steps_per_sec,
            "progress/total_steps": total_steps,
            "progress/updates": n_updates,
            "policy/std": current_std,
        }

        # Early stop on NaN/inf to avoid corrupting checkpoints/logs
        bad_keys = []
        for k, v in metrics.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                bad_keys.append(k)
        if bad_keys:
            print("\nEarly stop: NaN/inf detected in metrics: " + ", ".join(bad_keys))
            print(f"Stopped at iteration {iteration}, total_steps={total_steps:,}")
            break
        logger.log(metrics, iteration)
        
        # Evaluation
        if iteration % eval_interval == 0:
            print(f"\n--- Evaluation at iteration {iteration} ---")
            eval_stats = evaluate_policy(
                env=env,
                policy=policy,
                n_episodes=eval_episodes,
                deterministic=eval_deterministic,
            )
            
            for k, v in eval_stats.items():
                print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
            
            logger.log(eval_stats, iteration)
            
            # Save best model
            if eval_stats["eval/reward_mean"] > best_eval_reward:
                best_eval_reward = eval_stats["eval/reward_mean"]
                save_path = os.path.join(save_dir, "best_policy.pt")
                torch.save({
                    "policy_state_dict": policy.state_dict(),
                    "value_state_dict": value_net.state_dict() if value_net else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": iteration,
                    "total_steps": total_steps,
                    "eval_reward": best_eval_reward,
                }, save_path)
                print(f"  New best model saved! Reward: {best_eval_reward:.2f}")
            
            print()

            # Adaptive LR: increase when reward plateaus
            if adaptive_lr:
                current_eval = float(eval_stats["eval/reward_mean"])
                if last_eval_reward is None:
                    last_eval_reward = current_eval
                    eval_plateau_count = 0
                else:
                    if current_eval <= last_eval_reward + float(adaptive_lr_min_delta):
                        eval_plateau_count += 1
                    else:
                        eval_plateau_count = 0
                    last_eval_reward = current_eval

                if eval_plateau_count >= int(adaptive_lr_patience):
                    for opt in [optimizer, value_optimizer]:
                        if opt is None:
                            continue
                        for group in opt.param_groups:
                            new_lr = min(group["lr"] * float(adaptive_lr_factor), float(adaptive_lr_max))
                            group["lr"] = new_lr
                    eval_plateau_count = 0
                    print(f"  Adaptive LR: increased to {optimizer.param_groups[0]['lr']:.6f}")
        
        # Periodic checkpointing
        if iteration % save_interval == 0:
            save_path = os.path.join(save_dir, f"checkpoint_{iteration}.pt")
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "value_state_dict": value_net.state_dict() if value_net else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "value_optimizer_state_dict": value_optimizer.state_dict() if value_optimizer else None,
                "iteration": iteration,
                "total_steps": total_steps,
            }, save_path)
            print(f"Checkpoint saved: {save_path}")
    
    # Final save
    save_path = os.path.join(save_dir, "final_policy.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value_net.state_dict() if value_net else None,
        "iteration": total_iterations,
        "total_steps": total_steps,
    }, save_path)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Total timesteps: {total_steps:,}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"Final model saved: {save_path}")
    print(f"{'='*60}")
    
    logger.close()
    
    return policy, value_net


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CUDA-Parallelized PPO Training for Turret Aiming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Environment
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--n-envs", type=int, default=128,
                          help="Number of parallel environments")
    env_group.add_argument("--max-steps", type=int, default=500,
                          help="Max steps per episode")
    env_group.add_argument("--dt", type=float, default=0.015,
                          help="Simulation timestep (seconds)")
    
    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--total-timesteps", type=int, default=50_000_000,
                            help="Total environment steps")
    train_group.add_argument("--n-steps", type=int, default=128,
                            help="Steps per rollout before update")
    train_group.add_argument("--batch-size", type=int, default=128,
                            help="Mini-batch size for updates")
    train_group.add_argument("--n-epochs", type=int, default=2,
                            help="PPO epochs per update")
    train_group.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate")
    train_group.add_argument("--adaptive-lr", action="store_true", default=False,
                            help="Increase LR when eval reward plateaus")
    train_group.add_argument("--adaptive-lr-factor", type=float, default=1.2,
                            help="LR multiply factor on plateau")
    train_group.add_argument("--adaptive-lr-patience", type=int, default=3,
                            help="Eval checks to wait before increasing LR")
    train_group.add_argument("--adaptive-lr-min-delta", type=float, default=1.0,
                            help="Min eval improvement to reset plateau")
    train_group.add_argument("--adaptive-lr-max", type=float, default=1e-3,
                            help="Max LR cap for adaptive increases")
    train_group.add_argument("--gamma", type=float, default=0.99,
                            help="Discount factor")
    train_group.add_argument("--gae-lambda", type=float, default=0.95,
                            help="GAE lambda")
    
    # PPO
    ppo_group = parser.add_argument_group("PPO")
    ppo_group.add_argument("--clip-coef", type=float, default=0.05,
                          help="PPO clipping coefficient")
    ppo_group.add_argument("--value-coef", type=float, default=0.5,
                          help="Value loss coefficient")
    ppo_group.add_argument("--entropy-coef", type=float, default=0.01,
                          help="Entropy bonus coefficient")
    ppo_group.add_argument("--max-grad-norm", type=float, default=0.5,
                          help="Max gradient norm for clipping")
    ppo_group.add_argument("--target-kl", type=float, default=0.02,
                          help="Target KL for early stopping (None to disable)")
    
    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--hidden-dim", type=int, default=16,
                            help="Hidden layer dimension")
    model_group.add_argument("--num-layers", type=int, default=16,
                            help="Number of recurrent layers")
    model_group.add_argument("--use-actor-critic", action="store_true", default=True,
                            help="Use combined actor-critic network")
    model_group.add_argument("--no-actor-critic", action="store_false", dest="use_actor_critic",
                            help="Use separate policy and value networks")
    
    # Exploration
    explore_group = parser.add_argument_group("Exploration")
    explore_group.add_argument("--std-init", type=float, default=0.2,
                              help="Initial policy std")
    explore_group.add_argument("--std-decay", type=float, default=0.9999,
                              help="Std decay factor per update")
    explore_group.add_argument("--std-min", type=float, default=0.1,
                              help="Minimum policy std")
    
    # Reward shaping
    reward_group = parser.add_argument_group("Reward Shaping")
    reward_group.add_argument("--shot-base-penalty", type=float, default=1.0,
                             help="Base penalty for firing")
    reward_group.add_argument("--shot-miss-penalty", type=float, default=30.0,
                             help="Miss penalty scale")
    reward_group.add_argument("--miss-penalty", type=float, default=30.0,
                             help="Static penalty per missed shot (should exceed hit reward)")
    reward_group.add_argument("--hit-reward", type=float, default=20.0,
                             help="Reward per hit")
    reward_group.add_argument("--predicted-yaw-slope", type=float, default=100.0,
                             help="Slope m for predicted yaw reward: r = m*(err - tolerance)")
    reward_group.add_argument("--predicted-yaw-tolerance", type=float, default=1.0,
                             help="Tolerance (deg) used as b in r = m*(err - tolerance)")
    
    # Evaluation
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--eval-interval", type=int, default=10,
                           help="Evaluate every N iterations")
    eval_group.add_argument("--eval-episodes", type=int, default=50,
                           help="Episodes per evaluation")
    eval_group.add_argument("--eval-deterministic", action="store_true", default=False,
                           help="Use deterministic actions during evaluation")
    eval_group.add_argument("--eval-stochastic", action="store_false", dest="eval_deterministic",
                           help="Use stochastic actions during evaluation (matches rollouts)")
    
    # Checkpointing
    save_group = parser.add_argument_group("Checkpointing")
    save_group.add_argument("--save-interval", type=int, default=50,
                           help="Save checkpoint every N iterations")
    save_group.add_argument("--save-dir", type=str, default="checkpoints",
                           help="Directory for checkpoints")
    save_group.add_argument("--resume", type=str, default=None,
                           help="Path to checkpoint to resume from")
    
    # Misc
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--seed", type=int, default=None,
                           help="Random seed")
    misc_group.add_argument("--no-amp", action="store_true",
                           help="Disable automatic mixed precision")
    misc_group.add_argument("--log-interval", type=int, default=1,
                           help="Print logs every N iterations")
    misc_group.add_argument("--debug", action="store_true",
                           help="Enable debug mode (fewer envs, verbose)")
    
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: str,
    policy: nn.Module,
    value_net: Optional[nn.Module],
    optimizer: optim.Optimizer,
    value_optimizer: Optional[optim.Optimizer],
) -> Dict:
    """Load a training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    
    policy.load_state_dict(checkpoint["policy_state_dict"])
    
    if value_net is not None and checkpoint.get("value_state_dict") is not None:
        value_net.load_state_dict(checkpoint["value_state_dict"])
    
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if value_optimizer is not None and checkpoint.get("value_optimizer_state_dict") is not None:
        value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
    
    print(f"Resumed from iteration {checkpoint.get('iteration', 0)}, "
          f"total_steps={checkpoint.get('total_steps', 0):,}")
    
    return checkpoint


# ============================================================
# BENCHMARK FUNCTION
# ============================================================

def benchmark(
    n_envs: int = 1024,
    n_steps: int = 1000,
    hidden_dim: int = 256,
):
    """
    Benchmark environment and model throughput.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")
    print(f"  Environments: {n_envs}")
    print(f"  Steps: {n_steps}")
    print("  Model: gru")
    print()
    
    # Create environment
    env = create_vectorized_env(n_envs=n_envs, device=device, auto_reset=True)
    
    # Create policy
    policy = create_actor_critic(
        obs_dim=env.obs_dim,
        hidden_dim=hidden_dim,
        action_dim=3,
        device=device,
    )
    policy.eval()
    
    # Warmup
    print("Warming up...")
    obs = env.reset()
    hidden = None
    
    for _ in range(100):
        with torch.no_grad():
            action, _, _, hidden = policy.get_action_and_value(obs, hidden)
        obs, _, done, _ = env.step(action)
        
        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            hidden[:, done_idx, :] = 0
    
    # Benchmark
    print("Benchmarking...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    
    start = time.perf_counter()
    
    obs = env.reset()
    hidden = None
    
    for step in range(n_steps):
        with torch.no_grad():
            action, _, _, hidden = policy.get_action_and_value(obs, hidden)
        obs, _, done, _ = env.step(action)
        
        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            hidden[:, done_idx, :] = 0
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start
    
    total_steps = n_envs * n_steps
    steps_per_sec = total_steps / elapsed
    
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Total steps:     {total_steps:,}")
    print(f"Elapsed time:    {elapsed:.3f}s")
    print(f"Steps/second:    {steps_per_sec:,.0f}")
    print(f"FPS per env:     {n_steps / elapsed:.1f}")
    print(f"{'='*50}")
    
    # Memory stats
    if device.type == "cuda":
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    # Debug mode overrides
    if args.debug:
        args.n_envs = 32
        args.total_timesteps = 100_000
        args.eval_interval = 5
        args.save_interval = 10
        args.log_interval = 1
        print("DEBUG MODE ENABLED")
        print(f"  n_envs: {args.n_envs}")
        print(f"  total_timesteps: {args.total_timesteps}")
        print()
    
    # Print configuration
    print("="*60)
    print("CUDA-PARALLELIZED PPO TRAINING")
    print("="*60)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60)
    print()
    
    # Run training
    train(
        # Environment
        n_envs=args.n_envs,
        max_steps_per_episode=args.max_steps,
        dt=args.dt,
        # Training
        total_timesteps=args.total_timesteps,
        n_steps_per_rollout=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        adaptive_lr=args.adaptive_lr,
        adaptive_lr_factor=args.adaptive_lr_factor,
        adaptive_lr_patience=args.adaptive_lr_patience,
        adaptive_lr_min_delta=args.adaptive_lr_min_delta,
        adaptive_lr_max=args.adaptive_lr_max,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        # PPO
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        # Model
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_actor_critic=args.use_actor_critic,
        # Exploration
        policy_std_init=args.std_init,
        policy_std_decay=args.std_decay,
        policy_std_min=args.std_min,
        # Evaluation
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_deterministic=args.eval_deterministic,
        # Checkpointing
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        resume=args.resume,
        # Misc
        seed=args.seed,
        use_amp=not args.no_amp,
        log_interval=args.log_interval,
        # Reward shaping
        shot_base_penalty=args.shot_base_penalty,
        shot_miss_penalty_scale=args.shot_miss_penalty,
        miss_penalty=args.miss_penalty,
        hit_reward=args.hit_reward,
        predicted_yaw_slope=args.predicted_yaw_slope,
        predicted_yaw_tolerance_deg=args.predicted_yaw_tolerance,
    )


if __name__ == "__main__":
    import sys
    
    # Check for benchmark mode
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Remove 'benchmark' from args and parse rest
        sys.argv.pop(1)
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--n-envs", type=int, default=1024)
        parser.add_argument("--n-steps", type=int, default=1000)
        parser.add_argument("--hidden-dim", type=int, default=256)
        args = parser.parse_args()
        
        benchmark(
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            hidden_dim=args.hidden_dim,
        )
    else:
        main()
