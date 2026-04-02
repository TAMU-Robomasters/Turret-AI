"""
Vectorized CUDA Environment - Gym-like wrapper around VectorizedSimulator.

Provides batched reset/step API with reward computation, all on GPU.
"""

import torch
from typing import Optional, Dict, Tuple
import math

from vectorized_sim_cuda import VectorizedSimulator, WIDTH, HEIGHT, NUM_PANELS


class VectorizedTurretEnv:
    """
    Batched RL environment for turret aiming task.
    
    All N environments run in parallel on GPU.
    
    API:
        obs = env.reset()                    # (N, obs_dim)
        obs, reward, done, info = env.step(actions)  # actions: (N, 2) or (N, 3)
    """
    
    def __init__(
        self,
        n_envs: int,
        device: torch.device,
        dt: float = 0.015,
        max_steps: int = 500,
        obs_history_k: int = 4,
        # Action mode
        action_is_correction: bool = True,
        correction_baseline: str = "panel",
        correction_clip_yaw: float = math.radians(30.0),
        correction_clip_pitch: float = math.radians(15.0),
        # Fire timer
        maintain_fire_timer: bool = True,
        # Lost target penalties
        lost_done_steps: int = 150,
        lost_penalty_base: float = -200.0,
        lost_penalty_slope: float = -20.0,
        lost_penalty_cap_steps: int = 50,
        lost_terminal_penalty: float = -5000.0,
        # Shot penalties
        shot_base_penalty: float = 0.0,
        shot_miss_penalty_scale: float = 0.0,
        shot_miss_target_radius_mm: float = 60.0,
        shot_miss_penalty_power: float = 1.0,
        shot_miss_max_ratio: float = 5.0,
        shot_miss_max_angle_deg: float = 45.0,
        # Predicted-yaw reward (mx + b)
        predicted_yaw_slope: float = 10.0,
        predicted_yaw_tolerance_deg: float = 2.0,
        # Miss penalty (static per missed shot)
        miss_penalty: float = 150.0,
        # Tracking reward
        track_visible_reward: float = 0.0,
        # Motion penalties
        motion_penalty_coeff: float = 0.0,
        jerk_penalty_coeff: float = 5.0,
        # Hit reward
        hit_reward: float = 200.0,
        blind_fire_penalty: float = -200.0,
        # Dtype
        dtype: torch.dtype = torch.float32,
        # Spin behavior
        spin_mode: str = "constant",
        spin_rate: float = 9.0,
        spin_rate_std: float = 4.0,
        spin_noise_std: float = 0.02,
    ):
        self.n_envs = n_envs
        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.max_steps = max_steps
        
        # Action configuration
        self.action_is_correction = action_is_correction
        self.correction_baseline = correction_baseline
        self.correction_clip_yaw = correction_clip_yaw
        self.correction_clip_pitch = correction_clip_pitch
        self.maintain_fire_timer = maintain_fire_timer
        
        # Reward parameters
        self.lost_done_steps = lost_done_steps
        self.lost_penalty_base = lost_penalty_base
        self.lost_penalty_slope = lost_penalty_slope
        self.lost_penalty_cap_steps = lost_penalty_cap_steps
        self.lost_terminal_penalty = lost_terminal_penalty
        
        self.shot_base_penalty = shot_base_penalty
        self.shot_miss_penalty_scale = shot_miss_penalty_scale
        self.shot_miss_target_radius_mm = shot_miss_target_radius_mm
        self.shot_miss_penalty_power = shot_miss_penalty_power
        self.shot_miss_max_ratio = shot_miss_max_ratio
        self.shot_miss_max_angle_deg = math.radians(shot_miss_max_angle_deg)
        
        # Linear slope for predicted yaw reward: r = m*(err - tolerance)
        self.predicted_yaw_slope = predicted_yaw_slope
        self.predicted_yaw_tolerance_rad = math.radians(predicted_yaw_tolerance_deg)
        self.miss_penalty = miss_penalty
        
        self.track_visible_reward = track_visible_reward
        self.motion_penalty_coeff = motion_penalty_coeff
        self.jerk_penalty_coeff = jerk_penalty_coeff
        self.hit_reward = hit_reward
        self.blind_fire_penalty = blind_fire_penalty
        
        # Normalization constants
        self.max_distance = math.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT)
        self.speed_scale = 23000.0
        
        # Observation dimension (yaw-only features)
        self.base_obs_dim = 4
        self.obs_history_k = int(obs_history_k)
        if self.obs_history_k < 1:
            raise ValueError(f"obs_history_k must be >= 1, got {self.obs_history_k}")
        self.obs_dim = self.base_obs_dim * self.obs_history_k
        
        # Create simulator
        self.sim = VectorizedSimulator(
            n_envs=n_envs,
            device=device,
            dtype=dtype,
            spin_mode=spin_mode,
            spin_rate=spin_rate,
            spin_rate_std=spin_rate_std,
            spin_noise_std=spin_noise_std,
        )
        
        # Environment state tracking (N,)
        self.steps = torch.zeros(n_envs, device=device, dtype=torch.int32)
        self.lost_steps = torch.zeros(n_envs, device=device, dtype=torch.int32)
        self.prev_hit_count = torch.zeros(n_envs, device=device, dtype=torch.int32)
        self.prev_shots_fired = torch.zeros(n_envs, device=device, dtype=torch.int32)
        
        # Fire timer state (N,)
        self.time_to_fire_remaining = torch.full(
            (n_envs,), float('inf'), device=device, dtype=dtype
        )
        self.fire_timer_armed = torch.zeros(n_envs, device=device, dtype=torch.bool)
        
        # Previous camera deltas for jerk calculation (N,)
        self.prev_d_theta = torch.zeros(n_envs, device=device, dtype=dtype)
        self.prev_d_pitch = torch.zeros(n_envs, device=device, dtype=dtype)
        
        # dt tensor
        self._dt_tensor = torch.full((n_envs,), dt, device=device, dtype=dtype)

        # Observation history buffer: (N, K, base_obs_dim)
        self.obs_history = torch.zeros(
            n_envs, self.obs_history_k, self.base_obs_dim, device=device, dtype=dtype
        )

    def reset(
        self, 
        env_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reset environments.
        
        Args:
            env_mask: (N,) bool tensor, True = reset this env. None = reset all.
            seed: Random seed (applied globally, not per-env)
            
        Returns:
            obs: (N, obs_dim) or (n_reset, obs_dim) observations
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        if env_mask is None:
            env_mask = torch.ones(self.n_envs, device=self.device, dtype=torch.bool)
        
        # Reset simulator
        self.sim.reset(env_mask)
        
        # Reset environment tracking state
        self.steps[env_mask] = 0
        self.lost_steps[env_mask] = 0
        self.prev_hit_count[env_mask] = 0
        self.prev_shots_fired[env_mask] = 0
        self.time_to_fire_remaining[env_mask] = float('inf')
        self.fire_timer_armed[env_mask] = False
        self.prev_d_theta[env_mask] = 0.0
        self.prev_d_pitch[env_mask] = 0.0
        
        obs_base = self._get_obs_base()
        self._reset_history(obs_base, env_mask)
        return self._get_obs()

    def _get_obs_base(self) -> torch.Tensor:
        """Get normalized base observations from simulator."""
        return self.sim.get_model_input()  # Already normalized (N, 4)

    def _reset_history(self, obs_base: torch.Tensor, env_mask: torch.Tensor) -> None:
        """Reset history for selected envs using the current obs."""
        if env_mask is None:
            env_mask = torch.ones(self.n_envs, device=self.device, dtype=torch.bool)
        if env_mask.any():
            obs_repeat = obs_base[env_mask].unsqueeze(1).repeat(1, self.obs_history_k, 1)
            self.obs_history[env_mask] = obs_repeat

    def _update_history(self, obs_base: torch.Tensor) -> None:
        """Append current obs to history for all envs."""
        if self.obs_history_k == 1:
            self.obs_history[:, 0, :] = obs_base
            return
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = obs_base

    def _get_obs(self) -> torch.Tensor:
        """Return stacked observation history (oldest -> newest)."""
        return self.obs_history.reshape(self.n_envs, -1)

    def _get_baseline_aim(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get baseline aim direction (yaw, pitch) to predicted line (ideal lead).

        Returns:
            baseline_yaw: (N,)
            baseline_pitch: (N,)
        """
        # Use predicted line (ideal lead) from simulator
        baseline_yaw = self.sim.ideal_yaw
        baseline_pitch = self.sim.ideal_pitch

        return baseline_yaw, baseline_pitch
    def step(
        self, 
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Step all environments with given actions.
        
        Args:
            actions: (N, 2) or (N, 3) tensor
                - actions[:, 0]: yaw correction/delta/absolute
                - actions[:, 1]: time_to_fire command (if shape is (N, 2))
                - actions[:, 1]: pitch correction/delta/absolute (if shape is (N, 3), ignored)
                - actions[:, 2]: time_to_fire command (if shape is (N, 3))
                
        Returns:
            obs: (N, obs_dim) observations
            rewards: (N,) rewards
            dones: (N,) bool done flags
            info: dict of (N,) tensors with extra info
        """
        actions = actions.to(device=self.device, dtype=self.dtype)
        
        if actions.shape not in {(self.n_envs, 3), (self.n_envs, 2)}:
            raise ValueError(
                f"Expected actions shape ({self.n_envs}, 3) or ({self.n_envs}, 2), got {actions.shape}"
            )
        
        # === Parse Actions ===
        a0_raw = actions[:, 0]  # yaw
        if actions.shape[1] == 3:
            a1_raw = actions[:, 1]  # pitch (ignored for yaw-only control)
            time_to_fire_cmd = actions[:, 2]
        else:
            a1_raw = torch.zeros_like(a0_raw)
            time_to_fire_cmd = actions[:, 1]
        
        # === Compute Target Yaw/Pitch ===
        if self.action_is_correction:
            # Clip corrections (yaw only)
            a0 = torch.clamp(a0_raw, -self.correction_clip_yaw, self.correction_clip_yaw)

            # Get baseline (gravity-compensated ideal lead)
            baseline_yaw, baseline_pitch = self._get_baseline_aim()

            target_yaw = baseline_yaw + a0
            target_pitch = baseline_pitch
        else:
            # Absolute or delta mode
            target_yaw = a0_raw
            target_pitch = a1_raw
        
        # === Handle Fire Timer ===
        if self.maintain_fire_timer:
            # Arm timer if not already armed
            newly_armed = ~self.fire_timer_armed
            self.time_to_fire_remaining = torch.where(
                newly_armed,
                time_to_fire_cmd,
                self.time_to_fire_remaining
            )
            self.fire_timer_armed = self.fire_timer_armed | (time_to_fire_cmd < float('inf'))
            
            # Countdown
            self.time_to_fire_remaining = self.time_to_fire_remaining - self.dt
            
            # Determine firing
            fire_mask = self.time_to_fire_remaining <= 0.0
        else:
            # Interpret as time-until-fire this step; fire if within this step.
            fire_mask = time_to_fire_cmd <= self.dt
        
        # === Convert to Deltas ===
        d_theta = self._angle_diff(target_yaw, self.sim.camera_theta)
        d_pitch = target_pitch - self.sim.camera_pitch
        
        # === Store Pre-Step State for Reward ===
        prev_hit_count = self.sim.hit_count.clone()
        prev_shots_fired = self.sim.shots_fired.clone()
        
        # === Step Simulator ===
        step_info = self.sim.step(self._dt_tensor, d_theta, d_pitch, fire_mask)
        
        # === Reset Fire Timer After Firing ===
        if self.maintain_fire_timer:
            fired = self.sim.shots_fired > prev_shots_fired
            self.time_to_fire_remaining = torch.where(
                fired,
                torch.full_like(self.time_to_fire_remaining, float('inf')),
                self.time_to_fire_remaining
            )
            self.fire_timer_armed = self.fire_timer_armed & ~fired
        
        # === Update Step Counter ===
        self.steps = self.steps + 1
        
        # === Compute Rewards ===
        rewards = self._compute_rewards(
            prev_hit_count=prev_hit_count,
            prev_shots_fired=prev_shots_fired,
            d_theta=d_theta,
            d_pitch=d_pitch,
        )
        
        # === Update Previous State ===
        self.prev_hit_count = self.sim.hit_count.clone()
        self.prev_shots_fired = self.sim.shots_fired.clone()
        self.prev_d_theta = d_theta.clone()
        self.prev_d_pitch = d_pitch.clone()
        
        # === Compute Done Flags ===
        dones = self._compute_dones(step_info)
        
        # === Get Observations ===
        obs_base = self._get_obs_base()
        self._update_history(obs_base)
        obs = self._get_obs()
        
        # === Build Info Dict ===
        info = {
            "hit_count": self.sim.hit_count,
            "shots_fired": self.sim.shots_fired,
            "total_time": self.sim.total_time,
            "lost_steps": self.lost_steps,
            "target_yaw": target_yaw,
            "target_pitch": target_pitch,
            "ideal_yaw": self.sim.ideal_yaw,
            "ideal_pitch": self.sim.ideal_pitch,
            "new_hits": step_info["new_hits"],
        }
        
        return obs, rewards, dones, info

    def _angle_diff(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute smallest signed angle difference from b to a."""
        return torch.atan2(torch.sin(a - b), torch.cos(a - b))

    def _compute_rewards(
        self,
        prev_hit_count: torch.Tensor,
        prev_shots_fired: torch.Tensor,
        d_theta: torch.Tensor,
        d_pitch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rewards for all environments (vectorized).

        Simplified reward: predicted-line tracking + hit reward + miss penalty.
        """
        device = self.device
        dtype = self.dtype
        N = self.n_envs

        # === Hit Reward ===
        hit_inc = (self.sim.hit_count - prev_hit_count).to(dtype)
        r_hit = self.hit_reward * hit_inc

        # === Miss Penalty (static per missed shot) ===
        shots_inc = (self.sim.shots_fired - prev_shots_fired).to(dtype)
        r_shot_miss = torch.zeros(N, device=device, dtype=dtype)
        miss_shots = torch.clamp(shots_inc - hit_inc, min=0.0)
        if self.miss_penalty != 0.0:
            r_shot_miss = -float(self.miss_penalty) * miss_shots

        # === Predicted Yaw Reward (yaw only) ===
        yaw_err = self._angle_diff(self.sim.ideal_yaw, self.sim.camera_theta)
        angle_err = torch.abs(yaw_err)
        # Reward only: linear from 0 at tolerance to +scale at perfect aim.
        in_ratio = (self.predicted_yaw_tolerance_rad - angle_err) / self.predicted_yaw_tolerance_rad
        in_ratio = torch.clamp(in_ratio, min=0.0, max=1.0)
        r_predicted_yaw = self.predicted_yaw_slope * in_ratio

        # === Blind Fire Penalty (shots when no panels visible) ===
        r_blind_fire = torch.zeros(N, device=device, dtype=dtype)
        if self.blind_fire_penalty != 0.0:
            no_visible = (self.sim.last_shot_visible_count <= 0).to(dtype)
            r_blind_fire = self.blind_fire_penalty * shots_inc * no_visible

        # === Jerk Penalty (yaw only) ===
        r_jerk = torch.zeros(N, device=device, dtype=dtype)
        if self.jerk_penalty_coeff != 0.0:
            yaw_jerk = torch.abs(d_theta - self.prev_d_theta)
            r_jerk = -self.jerk_penalty_coeff * yaw_jerk

        # === Total Reward ===
        total_reward = r_hit + r_shot_miss + r_predicted_yaw + r_blind_fire + r_jerk
        return total_reward

    def _compute_dones(
        self, 
        step_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute done flags for all environments.
        
        Returns:
            dones: (N,) bool tensor
        """
        # Max steps reached
        max_steps_done = self.steps >= self.max_steps
        
        # Robot out of bounds
        robot_oob = step_info["robot_out_of_bounds"]
        
        # Lost target for too long
        lost_done = self.lost_steps >= self.lost_done_steps
        
        dones = max_steps_done | robot_oob | lost_done
        
        return dones

    def auto_reset(
        self, 
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Automatically reset environments that are done.
        Call this after step() if you want auto-reset behavior.
        
        Args:
            dones: (N,) bool tensor from step()
            
        Returns:
            obs: (N, obs_dim) observations (with reset envs having fresh obs)
        """
        if dones.any():
            self.reset(env_mask=dones)
        return self._get_obs()

    def get_episode_stats(self) -> Dict[str, torch.Tensor]:
        """Get current episode statistics for all environments."""
        return {
            "steps": self.steps,
            "hit_count": self.sim.hit_count,
            "shots_fired": self.sim.shots_fired,
            "total_time": self.sim.total_time,
            "lost_steps": self.lost_steps,
        }


class VectorizedTurretEnvWithAutoReset(VectorizedTurretEnv):
    """
    Vectorized environment with automatic reset on done.
    
    Also tracks episode returns and lengths for logging.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Episode tracking
        self.episode_returns = torch.zeros(self.n_envs, device=self.device, dtype=self.dtype)
        self.episode_lengths = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        
        # Completed episode stats (circular buffer per env, stores last completed)
        self.last_episode_return = torch.zeros(self.n_envs, device=self.device, dtype=self.dtype)
        self.last_episode_length = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.last_episode_hits = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.episodes_completed = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)

    def reset(
        self, 
        env_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Reset with episode tracking."""
        if env_mask is None:
            env_mask = torch.ones(self.n_envs, device=self.device, dtype=torch.bool)
        
        # Reset episode accumulators for reset envs
        self.episode_returns[env_mask] = 0.0
        self.episode_lengths[env_mask] = 0
        
        return super().reset(env_mask, seed)

    def step(
        self, 
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Step with auto-reset and episode tracking."""
        
        obs, rewards, dones, info = super().step(actions)
        
        # Accumulate episode stats
        self.episode_returns = self.episode_returns + rewards
        self.episode_lengths = self.episode_lengths + 1
        
        # Store completed episode stats before reset
        if dones.any():
            self.last_episode_return[dones] = self.episode_returns[dones]
            self.last_episode_length[dones] = self.episode_lengths[dones]
            self.last_episode_hits[dones] = self.sim.hit_count[dones]
            self.episodes_completed[dones] = self.episodes_completed[dones] + 1
            
            # Add to info
            info["episode_return"] = self.last_episode_return.clone()
            info["episode_length"] = self.last_episode_length.clone()
            info["episode_hits"] = self.last_episode_hits.clone()
            info["episode_done"] = dones.clone()
        
        # Auto-reset done environments
        if dones.any():
            # Get fresh observations for reset envs
            obs_after_reset = self.auto_reset(dones)
            # Replace observations for reset envs
            obs = torch.where(dones.unsqueeze(1), obs_after_reset, obs)
        
        return obs, rewards, dones, info

    def get_completed_episode_stats(self) -> Dict[str, torch.Tensor]:
        """Get stats from last completed episode per environment."""
        return {
            "return": self.last_episode_return,
            "length": self.last_episode_length,
            "hits": self.last_episode_hits,
            "episodes_completed": self.episodes_completed,
        }


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_vectorized_env(
    n_envs: int,
    device: Optional[torch.device] = None,
    auto_reset: bool = True,
    **kwargs,
) -> VectorizedTurretEnv:
    """
    Create a vectorized turret environment.
    
    Args:
        n_envs: Number of parallel environments
        device: PyTorch device (default: cuda if available)
        auto_reset: If True, use auto-reset wrapper
        **kwargs: Additional arguments to VectorizedTurretEnv
        
    Returns:
        VectorizedTurretEnv or VectorizedTurretEnvWithAutoReset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    EnvClass = VectorizedTurretEnvWithAutoReset if auto_reset else VectorizedTurretEnv
    
    env = EnvClass(n_envs=n_envs, device=device, **kwargs)
    env.reset()
    
    return env


# ============================================================
# TESTING / EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import time
    
    # Configuration
    N_ENVS = 1024
    N_STEPS = 1000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = create_vectorized_env(
        n_envs=N_ENVS,
        device=device,
        auto_reset=True,
    )
    
    print(f"Created {N_ENVS} parallel environments")
    print(f"Observation shape: ({N_ENVS}, {env.obs_dim})")
    
    # Reset
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    # Warm-up
    for _ in range(10):
        actions = torch.randn(N_ENVS, 3, device=device) * 0.1
        obs, rewards, dones, info = env.step(actions)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    
    total_rewards = torch.zeros(N_ENVS, device=device)
    
    for step in range(N_STEPS):
        # Random actions
        actions = torch.randn(N_ENVS, 3, device=device) * 0.1
        actions[:, 2] = torch.rand(N_ENVS, device=device)  # time_to_fire in [0, 1]
        
        obs, rewards, dones, info = env.step(actions)
        total_rewards = total_rewards + rewards
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start
    
    total_steps = N_ENVS * N_STEPS
    steps_per_sec = total_steps / elapsed
    
    print(f"\n=== Benchmark Results ===")
    print(f"Total steps: {total_steps:,}")
    print(f"Elapsed time: {elapsed:.3f}s")
    print(f"Steps/second: {steps_per_sec:,.0f}")
    print(f"Envs * FPS: {N_ENVS} * {N_STEPS/elapsed:.1f} = {steps_per_sec:,.0f}")
    
    # Episode stats
    stats = env.get_completed_episode_stats()
    completed = stats["episodes_completed"].sum().item()
    if completed > 0:
        mean_return = stats["return"][stats["episodes_completed"] > 0].mean().item()
        mean_length = stats["length"][stats["episodes_completed"] > 0].float().mean().item()
        mean_hits = stats["hits"][stats["episodes_completed"] > 0].float().mean().item()
        print(f"\n=== Episode Stats ({completed} episodes) ===")
        print(f"Mean return: {mean_return:.2f}")
        print(f"Mean length: {mean_length:.1f}")
        print(f"Mean hits: {mean_hits:.2f}")
