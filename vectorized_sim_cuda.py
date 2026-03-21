"""
Vectorized CUDA Simulator - runs N parallel environments on GPU.
All state is stored as PyTorch tensors with shape (N, ...) where N is batch size.
"""

import torch
from typing import Optional, Tuple, Dict
import math

# Constants
WIDTH = 12000.0
HEIGHT = 12000.0
GRAVITY = -9800.0
MAX_VEL = 4000.0
MAX_OMEGA = 6.0
MAX_OMEGA_ACC = 25.0
OMEGA_ACC_TAU = 0.6
OMEGA_ACC_SIGMA = 12.0
NUM_PANELS = 4


class VectorizedSimulator:
    """
    Batched simulator running N environments in parallel on GPU.
    All tensors have shape (N, ...) where N is the batch/environment dimension.
    """
    
    def __init__(
        self,
        n_envs: int,
        device: torch.device,
        projectile_speed: float = 23000.0,
        max_projectiles: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_envs = n_envs
        self.device = device
        self.dtype = dtype
        self.projectile_speed = projectile_speed
        self.max_projectiles = max_projectiles
        
        # Pre-compute constants on device
        self._panel_base_thetas = torch.tensor(
            [0.0, math.pi, math.pi/2, 3*math.pi/2], 
            device=device, dtype=dtype
        )
        self._gravity = torch.tensor(GRAVITY, device=device, dtype=dtype)
        self._width = torch.tensor(WIDTH, device=device, dtype=dtype)
        self._height = torch.tensor(HEIGHT, device=device, dtype=dtype)
        
        # Initialize all state tensors
        self._init_state()
        
    def _init_state(self):
        """Initialize all state tensors to zeros/defaults."""
        N, device, dtype = self.n_envs, self.device, self.dtype
        
        # === Camera state (N,) ===
        self.camera_x = torch.full((N,), WIDTH / 2, device=device, dtype=dtype)
        self.camera_y = torch.full((N,), HEIGHT / 2, device=device, dtype=dtype)
        self.camera_z = torch.zeros(N, device=device, dtype=dtype)
        self.camera_theta = torch.full((N,), -math.pi / 2, device=device, dtype=dtype)
        self.camera_pitch = torch.zeros(N, device=device, dtype=dtype)
        self.camera_fov = torch.full((N,), math.pi / 3, device=device, dtype=dtype)
        
        # === Robot state (N,) ===
        self.robot_radius = torch.zeros(N, device=device, dtype=dtype)
        self.robot_x = torch.zeros(N, device=device, dtype=dtype)
        self.robot_y = torch.zeros(N, device=device, dtype=dtype)
        self.robot_z = torch.zeros(N, device=device, dtype=dtype)
        self.robot_vx = torch.zeros(N, device=device, dtype=dtype)
        self.robot_vy = torch.zeros(N, device=device, dtype=dtype)
        self.robot_ax = torch.zeros(N, device=device, dtype=dtype)
        self.robot_ay = torch.zeros(N, device=device, dtype=dtype)
        self.robot_theta = torch.zeros(N, device=device, dtype=dtype)
        self.robot_omega = torch.zeros(N, device=device, dtype=dtype)
        self.robot_omega_acc = torch.zeros(N, device=device, dtype=dtype)
        
        # === Panel state (N, NUM_PANELS) ===
        self.panel_x = torch.zeros(N, NUM_PANELS, device=device, dtype=dtype)
        self.panel_y = torch.zeros(N, NUM_PANELS, device=device, dtype=dtype)
        self.panel_z = torch.zeros(N, NUM_PANELS, device=device, dtype=dtype)
        self.panel_z_offset = torch.zeros(N, NUM_PANELS, device=device, dtype=dtype)
        self.panel_visible = torch.zeros(N, NUM_PANELS, device=device, dtype=torch.bool)
        
        # === Projectile state (N, max_projectiles) ===
        self.proj_x = torch.zeros(N, self.max_projectiles, device=device, dtype=dtype)
        self.proj_y = torch.zeros(N, self.max_projectiles, device=device, dtype=dtype)
        self.proj_z = torch.zeros(N, self.max_projectiles, device=device, dtype=dtype)
        self.proj_vx = torch.zeros(N, self.max_projectiles, device=device, dtype=dtype)
        self.proj_vy = torch.zeros(N, self.max_projectiles, device=device, dtype=dtype)
        self.proj_vz = torch.zeros(N, self.max_projectiles, device=device, dtype=dtype)
        self.proj_alive = torch.zeros(N, self.max_projectiles, device=device, dtype=torch.bool)
        self.proj_allowed_panels = torch.zeros(
            N, self.max_projectiles, NUM_PANELS, device=device, dtype=torch.bool
        )
        
        # === Metrics (N,) ===
        self.total_time = torch.zeros(N, device=device, dtype=dtype)
        self.hit_count = torch.zeros(N, device=device, dtype=torch.int32)
        self.shots_fired = torch.zeros(N, device=device, dtype=torch.int32)
        self.last_shot_visible_count = torch.zeros(N, device=device, dtype=torch.int32)
        
        # === Velocity estimation (N, 3) ===
        self.prev_rel_center = torch.zeros(N, 3, device=device, dtype=dtype)
        self.rel_velocity = torch.zeros(N, 3, device=device, dtype=dtype)
        
        # === Ideal lead angles (N,) ===
        self.ideal_yaw = torch.zeros(N, device=device, dtype=dtype)
        self.ideal_pitch = torch.zeros(N, device=device, dtype=dtype)
        
        # === Last camera deltas (N,) ===
        self.last_d_theta = torch.zeros(N, device=device, dtype=dtype)
        self.last_d_pitch = torch.zeros(N, device=device, dtype=dtype)

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """
        Reset environments. 
        env_mask: (N,) bool tensor, True = reset this env. None = reset all.
        """
        if env_mask is None:
            env_mask = torch.ones(self.n_envs, device=self.device, dtype=torch.bool)
        
        n_reset = env_mask.sum().item()
        if n_reset == 0:
            return
            
        device, dtype = self.device, self.dtype
        
        # Reset camera orientation
        self.camera_theta[env_mask] = -math.pi / 2
        self.camera_pitch[env_mask] = 0.0
        
        # Random robot radius
        self.robot_radius[env_mask] = (
            torch.randint(150, 251, (n_reset,), device=device).to(dtype)
        )
        
        # Random position away from camera (rejection sampling)
        min_dist = 300.0
        rand_x = torch.rand(n_reset, device=device, dtype=dtype) * (WIDTH - 500) + 250
        rand_y = torch.rand(n_reset, device=device, dtype=dtype) * (HEIGHT - 500) + 250
        
        for _ in range(10):
            dx = rand_x - WIDTH / 2
            dy = rand_y - HEIGHT / 2
            dist = torch.sqrt(dx * dx + dy * dy)
            bad = dist < min_dist
            if not bad.any():
                break
            rand_x[bad] = torch.rand(bad.sum().item(), device=device, dtype=dtype) * (WIDTH - 500) + 250
            rand_y[bad] = torch.rand(bad.sum().item(), device=device, dtype=dtype) * (HEIGHT - 500) + 250
        
        self.robot_x[env_mask] = rand_x
        self.robot_y[env_mask] = rand_y
        self.robot_z[env_mask] = 0.0
        
        # Random initial velocities
        self.robot_vx[env_mask] = (torch.rand(n_reset, device=device, dtype=dtype) - 0.5) * MAX_VEL
        self.robot_vy[env_mask] = (torch.rand(n_reset, device=device, dtype=dtype) - 0.5) * MAX_VEL
        self.robot_ax[env_mask] = 0.0
        self.robot_ay[env_mask] = 0.0
        
        # Random initial angular state
        self.robot_theta[env_mask] = 0.0
        self.robot_omega[env_mask] = (torch.rand(n_reset, device=device, dtype=dtype) - 0.5) * MAX_OMEGA
        self.robot_omega_acc[env_mask] = (torch.rand(n_reset, device=device, dtype=dtype) - 0.5) * 0.5 * MAX_OMEGA_ACC
        
        # Panel z offsets
        h1 = torch.rand(n_reset, device=device, dtype=dtype) * (-20) - 10
        h2 = torch.rand(n_reset, device=device, dtype=dtype) * (-20) - 10
        self.panel_z_offset[env_mask, 0] = h1
        self.panel_z_offset[env_mask, 1] = h2
        self.panel_z_offset[env_mask, 2] = h1
        self.panel_z_offset[env_mask, 3] = h2
        
        # Clear projectiles
        self.proj_alive[env_mask] = False
        
        # Reset metrics
        self.total_time[env_mask] = 0.0
        self.hit_count[env_mask] = 0
        self.shots_fired[env_mask] = 0
        self.last_shot_visible_count[env_mask] = 0
        self.prev_rel_center[env_mask] = 0.0
        self.rel_velocity[env_mask] = 0.0
        self.last_d_theta[env_mask] = 0.0
        self.last_d_pitch[env_mask] = 0.0
        
        # Update panel positions
        self._update_panels()
        
        # Aim camera roughly at robot
        dx = self.robot_x - self.camera_x
        dy = self.robot_y - self.camera_y
        base_yaw = torch.atan2(dy, dx)
        yaw_offset = (torch.rand(self.n_envs, device=device, dtype=dtype) - 0.5) * self.camera_fov / 2
        self.camera_theta = torch.where(env_mask, base_yaw + yaw_offset, self.camera_theta)
        # ============================================================
    # PHYSICS UPDATE METHODS (add these to VectorizedSimulator class)
    # ============================================================
    
    def _update_panels(self) -> None:
        """Update panel positions and visibility for all environments vectorized."""
        # Panel angles in world frame: robot_theta + base_theta for each panel
        # Shape: (N, NUM_PANELS)
        panel_theta = self.robot_theta.unsqueeze(1) + self._panel_base_thetas.unsqueeze(0)
        
        # Panel world positions
        cos_theta = torch.cos(panel_theta)
        sin_theta = torch.sin(panel_theta)
        
        self.panel_x = self.robot_x.unsqueeze(1) + self.robot_radius.unsqueeze(1) * cos_theta
        self.panel_y = self.robot_y.unsqueeze(1) + self.robot_radius.unsqueeze(1) * sin_theta
        self.panel_z = self.robot_z.unsqueeze(1) + self.panel_z_offset
        
        # === Visibility Check ===
        # Vector from camera to panel
        dx = self.panel_x - self.camera_x.unsqueeze(1)  # (N, NUM_PANELS)
        dy = self.panel_y - self.camera_y.unsqueeze(1)
        
        # Angle from camera to panel
        angle_to_panel = torch.atan2(dy, dx)
        
        # Relative angle (normalized to [-pi, pi])
        relative_angle = angle_to_panel - self.camera_theta.unsqueeze(1)
        relative_angle = torch.atan2(torch.sin(relative_angle), torch.cos(relative_angle))
        
        # Check if in FOV
        in_fov = torch.abs(relative_angle) <= (self.camera_fov.unsqueeze(1) / 2)
        
        # Panel outward normal direction
        nx = cos_theta
        ny = sin_theta
        
        # Vector from panel to camera
        vx = self.camera_x.unsqueeze(1) - self.panel_x
        vy = self.camera_y.unsqueeze(1) - self.panel_y
        
        # Facing camera if dot product > 0
        facing_camera = (nx * vx + ny * vy) > 0
        
        # Visible = in FOV AND facing camera
        self.panel_visible = in_fov & facing_camera

    def _update_robot(self, dt: torch.Tensor) -> None:
        """Update robot physics for all environments vectorized."""
        # Ensure dt has correct shape
        if dt.dim() == 0:
            dt = dt.expand(self.n_envs)
        
        # === Random Maneuvers (5% chance per env) ===
        maneuver_rand = torch.rand(self.n_envs, device=self.device, dtype=self.dtype)
        maneuver_mask = maneuver_rand < 0.05
        n_maneuver = maneuver_mask.sum().item()
        
        if n_maneuver > 0:
            # Random acceleration changes
            self.robot_ax[maneuver_mask] = (
                torch.rand(n_maneuver, device=self.device, dtype=self.dtype) - 0.5
            ) * 1500.0
            self.robot_ay[maneuver_mask] = (
                torch.rand(n_maneuver, device=self.device, dtype=self.dtype) - 0.5
            ) * 1500.0
            # Random angular acceleration kick
            self.robot_omega_acc[maneuver_mask] = (
                torch.rand(n_maneuver, device=self.device, dtype=self.dtype) - 0.5
            ) * 2.0 * MAX_OMEGA_ACC
        
        # === Linear Velocity Update ===
        self.robot_vx = self.robot_vx + self.robot_ax * dt
        self.robot_vy = self.robot_vy + self.robot_ay * dt
        
        # Clamp velocities
        self.robot_vx = torch.clamp(self.robot_vx, -MAX_VEL, MAX_VEL)
        self.robot_vy = torch.clamp(self.robot_vy, -MAX_VEL, MAX_VEL)
        
        # === Position Update (with acceleration term) ===
        self.robot_x = self.robot_x + self.robot_vx * dt + 0.5 * self.robot_ax * dt * dt
        self.robot_y = self.robot_y + self.robot_vy * dt + 0.5 * self.robot_ay * dt * dt
        
        # === Angular Dynamics (Ornstein-Uhlenbeck process) ===
        decay = dt / OMEGA_ACC_TAU
        noise = torch.randn(self.n_envs, device=self.device, dtype=self.dtype)
        
        # OU update for angular acceleration
        self.robot_omega_acc = self.robot_omega_acc * (1.0 - decay)
        self.robot_omega_acc = self.robot_omega_acc + OMEGA_ACC_SIGMA * torch.sqrt(dt) * noise
        
        # Integrate angular velocity and position
        self.robot_omega = self.robot_omega + self.robot_omega_acc * dt
        self.robot_omega = torch.clamp(self.robot_omega, -MAX_OMEGA, MAX_OMEGA)
        self.robot_theta = self.robot_theta + self.robot_omega * dt
        
        # === Boundary Clamping ===
        self.robot_x = torch.clamp(self.robot_x, self.robot_radius, WIDTH - self.robot_radius)
        self.robot_y = torch.clamp(self.robot_y, self.robot_radius, HEIGHT - self.robot_radius)
        
        # === Avoid Camera ===
        self._avoid_camera(min_dist=300.0)
        
        # === Update Panel Positions ===
        self._update_panels()

    def _avoid_camera(self, min_dist: float = 300.0) -> None:
        """Push robot away from camera if too close (vectorized)."""
        dx = self.robot_x - self.camera_x
        dy = self.robot_y - self.camera_y
        dist = torch.sqrt(dx * dx + dy * dy)
        
        too_close = dist < min_dist
        if not too_close.any():
            return
        
        # Angle from camera to robot
        angle = torch.atan2(dy, dx)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Push position outward
        new_x = self.camera_x + min_dist * cos_angle
        new_y = self.camera_y + min_dist * sin_angle
        self.robot_x = torch.where(too_close, new_x, self.robot_x)
        self.robot_y = torch.where(too_close, new_y, self.robot_y)
        
        # Remove velocity component toward camera
        v_radial = self.robot_vx * cos_angle + self.robot_vy * sin_angle
        inward_velocity = too_close & (v_radial < 0)
        
        self.robot_vx = torch.where(
            inward_velocity,
            self.robot_vx - v_radial * cos_angle,
            self.robot_vx
        )
        self.robot_vy = torch.where(
            inward_velocity,
            self.robot_vy - v_radial * sin_angle,
            self.robot_vy
        )

    def _update_projectiles(self, dt: torch.Tensor, hit_tol_mm: float = 100.0) -> torch.Tensor:
        """
        Update all projectiles and check for hits (vectorized).
        
        Returns:
            new_hits: (N,) int tensor - number of new hits per environment this step
        """
        if dt.dim() == 0:
            dt = dt.expand(self.n_envs)
        
        # Expand dt for projectile dimension: (N,) -> (N, 1)
        dt_proj = dt.unsqueeze(1)
        
        # === Physics Update ===
        # Apply gravity to vertical velocity
        self.proj_vz = torch.where(
            self.proj_alive,
            self.proj_vz + self._gravity * dt_proj,
            self.proj_vz
        )
        
        # Update positions (only for alive projectiles, but we compute all for efficiency)
        self.proj_x = self.proj_x + self.proj_vx * dt_proj
        self.proj_y = self.proj_y + self.proj_vy * dt_proj
        self.proj_z = self.proj_z + self.proj_vz * dt_proj
        
        # === Kill Out-of-Bounds Projectiles ===
        out_of_bounds = (
            (self.proj_x < 0) | (self.proj_x > WIDTH) |
            (self.proj_y < 0) | (self.proj_y > HEIGHT) |
            (self.proj_z < -20000)
        )
        self.proj_alive = self.proj_alive & ~out_of_bounds
        
        # === Hit Detection ===
        # Compute distance from each projectile to each panel
        # proj: (N, max_proj), panel: (N, NUM_PANELS)
        # Expand: proj -> (N, max_proj, 1), panel -> (N, 1, NUM_PANELS)
        
        dx = self.proj_x.unsqueeze(2) - self.panel_x.unsqueeze(1)  # (N, max_proj, NUM_PANELS)
        dy = self.proj_y.unsqueeze(2) - self.panel_y.unsqueeze(1)
        dz = self.proj_z.unsqueeze(2) - self.panel_z.unsqueeze(1)
        
        dist_sq = dx * dx + dy * dy + dz * dz
        dist = torch.sqrt(dist_sq)
        
        # Check if within hit tolerance
        within_range = dist <= hit_tol_mm  # (N, max_proj, NUM_PANELS)
        
        # Check if panel was allowed for this projectile
        # proj_allowed_panels: (N, max_proj, NUM_PANELS)
        allowed = self.proj_allowed_panels
        
        # Hit = alive AND within_range AND allowed (for at least one panel)
        hit_panel = within_range & allowed & self.proj_alive.unsqueeze(2)  # (N, max_proj, NUM_PANELS)
        
        # A projectile hits if it hits ANY allowed panel
        proj_hit = hit_panel.any(dim=2)  # (N, max_proj)
        
        # Count new hits per environment
        new_hits = proj_hit.sum(dim=1).to(torch.int32)  # (N,)
        
        # Kill projectiles that hit
        self.proj_alive = self.proj_alive & ~proj_hit
        
        # Update hit count
        self.hit_count = self.hit_count + new_hits
        
        return new_hits

    def fire_projectiles(self, fire_mask: torch.Tensor) -> None:
        """
        Fire projectiles from specified environments.
        
        Args:
            fire_mask: (N,) bool tensor - True = fire a projectile from this env
        """
        if not fire_mask.any():
            return
        
        n_fire = fire_mask.sum().item()
        device, dtype = self.device, self.dtype
        
        # Find first free projectile slot for each firing environment
        # We'll use a simple approach: find first False in proj_alive per env
        
        for env_idx in torch.where(fire_mask)[0]:
            env_i = env_idx.item()
            
            # Find first available slot
            alive_slots = self.proj_alive[env_i]
            free_slots = ~alive_slots
            
            if not free_slots.any():
                # No free slots, skip (or could kill oldest)
                continue
            
            slot_idx = torch.where(free_slots)[0][0].item()
            
            # Get camera state for this env
            cam_x = self.camera_x[env_i]
            cam_y = self.camera_y[env_i]
            cam_z = self.camera_z[env_i]
            cam_theta = self.camera_theta[env_i]
            cam_pitch = self.camera_pitch[env_i]
            
            # Compute projectile velocity
            speed = self.projectile_speed
            horiz_speed = speed * torch.cos(cam_pitch)
            
            vx = horiz_speed * torch.cos(cam_theta)
            vy = horiz_speed * torch.sin(cam_theta)
            vz = speed * torch.sin(cam_pitch)
            
            # Initialize projectile
            self.proj_x[env_i, slot_idx] = cam_x
            self.proj_y[env_i, slot_idx] = cam_y
            self.proj_z[env_i, slot_idx] = cam_z
            self.proj_vx[env_i, slot_idx] = vx
            self.proj_vy[env_i, slot_idx] = vy
            self.proj_vz[env_i, slot_idx] = vz
            self.proj_alive[env_i, slot_idx] = True
            
            # Store which panels were visible at fire time
            self.proj_allowed_panels[env_i, slot_idx] = self.panel_visible[env_i]
            
            # Update metrics
            self.shots_fired[env_i] += 1
            self.last_shot_visible_count[env_i] = self.panel_visible[env_i].sum().to(torch.int32)

    def fire_projectiles_batched(self, fire_mask: torch.Tensor) -> None:
        """
        Fire projectiles from specified environments (fully vectorized version).
        
        Args:
            fire_mask: (N,) bool tensor - True = fire a projectile from this env
        """
        if not fire_mask.any():
            return
        
        # Find first free slot per environment
        # proj_alive: (N, max_proj)
        # We want the index of the first False per row
        
        # Create indices
        indices = torch.arange(self.max_projectiles, device=self.device)
        
        # Mask: set alive slots to max_proj (so they won't be selected as min)
        masked = torch.where(
            self.proj_alive,
            torch.full_like(self.proj_alive, self.max_projectiles, dtype=torch.long),
            indices.unsqueeze(0).expand(self.n_envs, -1)
        )
        
        # First free slot index per env
        first_free, _ = masked.min(dim=1)  # (N,)
        
        # Environments that can fire (have free slot AND want to fire)
        can_fire = fire_mask & (first_free < self.max_projectiles)
        
        if not can_fire.any():
            return
        
        # Get indices of envs that will fire
        fire_env_idx = torch.where(can_fire)[0]
        fire_slot_idx = first_free[can_fire]
        
        # Gather camera state for firing envs
        cam_x = self.camera_x[can_fire]
        cam_y = self.camera_y[can_fire]
        cam_z = self.camera_z[can_fire]
        cam_theta = self.camera_theta[can_fire]
        cam_pitch = self.camera_pitch[can_fire]
        
        # Compute velocities
        speed = self.projectile_speed
        horiz_speed = speed * torch.cos(cam_pitch)
        
        vx = horiz_speed * torch.cos(cam_theta)
        vy = horiz_speed * torch.sin(cam_theta)
        vz = speed * torch.sin(cam_pitch)
        
        # Scatter into projectile arrays using advanced indexing
        self.proj_x[fire_env_idx, fire_slot_idx] = cam_x
        self.proj_y[fire_env_idx, fire_slot_idx] = cam_y
        self.proj_z[fire_env_idx, fire_slot_idx] = cam_z
        self.proj_vx[fire_env_idx, fire_slot_idx] = vx
        self.proj_vy[fire_env_idx, fire_slot_idx] = vy
        self.proj_vz[fire_env_idx, fire_slot_idx] = vz
        self.proj_alive[fire_env_idx, fire_slot_idx] = True
        
        # Store allowed panels (visible at fire time)
        self.proj_allowed_panels[fire_env_idx, fire_slot_idx] = self.panel_visible[can_fire]
        
        # Update metrics
        self.shots_fired[can_fire] += 1
        self.last_shot_visible_count[can_fire] = self.panel_visible[can_fire].sum(dim=1).to(torch.int32)

        # ============================================================
    # STEP FUNCTION & OBSERVATIONS (add these to VectorizedSimulator class)
    # ============================================================

    def _compute_ideal_lead(self, dt: torch.Tensor) -> None:
        """
        Compute ideal lead angles (yaw and pitch) for hitting the robot center.
        Uses simple linear prediction with iterative refinement.
        """
        if dt.dim() == 0:
            dt = dt.expand(self.n_envs)
        
        # Relative position from camera to robot
        rel_x = self.robot_x - self.camera_x
        rel_y = self.robot_y - self.camera_y
        rel_z = self.robot_z - self.camera_z
        
        # Estimate velocity via finite differences
        current_rel = torch.stack([rel_x, rel_y, rel_z], dim=1)  # (N, 3)
        
        # Compute velocity (handle first step where prev is zero)
        valid_prev = self.prev_rel_center.abs().sum(dim=1) > 1e-6
        
        vel = torch.zeros_like(current_rel)
        vel[valid_prev] = (current_rel[valid_prev] - self.prev_rel_center[valid_prev]) / dt[valid_prev].unsqueeze(1)
        
        # Store for next step
        self.prev_rel_center = current_rel.clone()
        self.rel_velocity = vel
        
        # Initial time estimate
        dist = torch.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
        T = dist / self.projectile_speed
        
        # Iterative refinement (2 iterations)
        for _ in range(2):
            future_pos = current_rel + vel * T.unsqueeze(1)
            dist_future = torch.norm(future_pos, dim=1)
            T = dist_future / self.projectile_speed
        
        # Final predicted position
        future_pos = current_rel + vel * T.unsqueeze(1)
        fx, fy, fz = future_pos[:, 0], future_pos[:, 1], future_pos[:, 2]
        
        # Ideal yaw (horizontal angle)
        self.ideal_yaw = torch.atan2(fy, fx)
        
        # Ideal pitch (with ballistic correction)
        horiz_dist = torch.sqrt(fx * fx + fy * fy)
        
        # Ballistic pitch calculation
        self.ideal_pitch = self._ballistic_pitch_vectorized(horiz_dist, fz)

    def _ballistic_pitch_vectorized(
        self, 
        horiz_dist: torch.Tensor, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized ballistic pitch calculation.
        
        Solves for launch pitch to hit target at (horiz_dist, z) with gravity.
        
        Args:
            horiz_dist: (N,) horizontal distance to target
            z: (N,) vertical offset to target
            
        Returns:
            pitch: (N,) launch pitch angles in radians
        """
        speed = self.projectile_speed
        g_abs = -self._gravity  # positive value
        
        v2 = speed * speed
        R = horiz_dist.clamp(min=1e-6)
        
        # Discriminant: v^4 - g*(g*R^2 + 2*z*v^2)
        disc = v2 * v2 - g_abs * (g_abs * R * R + 2.0 * z * v2)
        
        # Where discriminant is negative, no solution exists - fall back to straight line
        valid = disc >= 0
        
        sqrt_disc = torch.sqrt(disc.clamp(min=0))
        denom = (g_abs * R).clamp(min=1e-12)
        
        # Two solutions
        tan1 = (v2 + sqrt_disc) / denom
        tan2 = (v2 - sqrt_disc) / denom
        
        pitch1 = torch.atan(tan1)
        pitch2 = torch.atan(tan2)
        
        # Choose smaller magnitude pitch (flatter trajectory)
        pitch = torch.where(torch.abs(pitch1) < torch.abs(pitch2), pitch1, pitch2)
        
        # Clamp to reasonable range
        max_pitch = math.radians(89)
        pitch = torch.clamp(pitch, -max_pitch, max_pitch)
        
        # Fall back to straight-line pitch where ballistic solution doesn't exist
        straight_pitch = torch.atan2(z, R)
        pitch = torch.where(valid, pitch, straight_pitch)
        
        return pitch

    def _get_target_panel_idx(self) -> torch.Tensor:
        """
        Get index of target panel for each environment.
        Prefers closest visible panel; falls back to closest overall.
        
        Returns:
            target_idx: (N,) int tensor of panel indices [0, NUM_PANELS)
        """
        # Distances to all panels: (N, NUM_PANELS)
        dx = self.panel_x - self.camera_x.unsqueeze(1)
        dy = self.panel_y - self.camera_y.unsqueeze(1)
        dz = self.panel_z - self.camera_z.unsqueeze(1)
        dist = torch.sqrt(dx * dx + dy * dy + dz * dz)
        
        # Mask non-visible panels with large distance for "closest visible" search
        large_dist = 1e9
        dist_visible = torch.where(self.panel_visible, dist, large_dist)
        
        # Find closest visible
        closest_visible_dist, closest_visible_idx = dist_visible.min(dim=1)
        
        # Find closest overall (for fallback)
        closest_any_dist, closest_any_idx = dist.min(dim=1)
        
        # Use visible if any visible, else use any
        has_visible = self.panel_visible.any(dim=1)
        target_idx = torch.where(has_visible, closest_visible_idx, closest_any_idx)
        
        return target_idx

    def get_model_input(self) -> torch.Tensor:
        """
        Get normalized observation features for all environments.
        
        Returns:
            obs: (N, 7) tensor of normalized features:
                [camera_yaw/pi, camera_pitch/pi, yaw_to_panel/pi, pitch_to_panel/pi,
                 panel_yaw_world/pi, distance_normalized, speed_normalized]
        """
        # Get target panel for each env
        target_idx = self._get_target_panel_idx()  # (N,)
        
        # Gather target panel positions using advanced indexing
        batch_idx = torch.arange(self.n_envs, device=self.device)
        
        target_x = self.panel_x[batch_idx, target_idx]
        target_y = self.panel_y[batch_idx, target_idx]
        target_z = self.panel_z[batch_idx, target_idx]
        
        # Relative position from camera to target panel
        rel_x = target_x - self.camera_x
        rel_y = target_y - self.camera_y
        rel_z = target_z - self.camera_z
        
        horiz_dist = torch.sqrt(rel_x * rel_x + rel_y * rel_y)
        distance = torch.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
        
        # Yaw/pitch to panel
        yaw_to_panel = torch.atan2(rel_y, rel_x)
        pitch_to_panel = torch.where(
            horiz_dist > 1e-6,
            torch.atan2(rel_z, horiz_dist),
            torch.zeros_like(rel_z)
        )
        
        # Panel outward normal yaw (world frame)
        panel_theta = self.robot_theta + self._panel_base_thetas[target_idx]
        panel_yaw_world = torch.atan2(torch.sin(panel_theta), torch.cos(panel_theta))
        
        # Normalization constants
        max_distance = math.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT)
        speed_scale = 23000.0
        
        # Build feature tensor
        obs = torch.stack([
            self.camera_theta / math.pi,
            self.camera_pitch / math.pi,
            yaw_to_panel / math.pi,
            pitch_to_panel / math.pi,
            panel_yaw_world / math.pi,
            distance / max_distance,
            torch.full((self.n_envs,), self.projectile_speed / speed_scale, 
                       device=self.device, dtype=self.dtype),
        ], dim=1)  # (N, 7)
        
        return obs

    def step(
        self,
        dt: torch.Tensor,
        d_theta: torch.Tensor,
        d_pitch: torch.Tensor,
        fire_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Advance simulation by one step for all environments.
        
        Args:
            dt: (N,) or scalar - time step per environment
            d_theta: (N,) - camera yaw delta
            d_pitch: (N,) - camera pitch delta
            fire_mask: (N,) bool - True = fire projectile this step
            
        Returns:
            info dict with:
                - new_hits: (N,) int tensor
                - robot_out_of_bounds: (N,) bool tensor
        """
        # Handle scalar dt
        if dt.dim() == 0:
            dt = dt.expand(self.n_envs)
        
        # Ensure tensors are on correct device
        d_theta = d_theta.to(device=self.device, dtype=self.dtype)
        d_pitch = d_pitch.to(device=self.device, dtype=self.dtype)
        
        # Store last deltas
        self.last_d_theta = d_theta.clone()
        self.last_d_pitch = d_pitch.clone()
        
        # === Update Camera ===
        self.camera_theta = self.camera_theta + d_theta
        self.camera_pitch = self.camera_pitch + d_pitch
        
        # Normalize yaw to [-pi, pi]
        self.camera_theta = torch.atan2(
            torch.sin(self.camera_theta),
            torch.cos(self.camera_theta)
        )
        
        # Clamp pitch
        max_pitch = math.radians(89)
        self.camera_pitch = torch.clamp(self.camera_pitch, -max_pitch, max_pitch)
        
        # === Update Robot ===
        self._update_robot(dt)
        
        # === Update Projectiles ===
        new_hits = self._update_projectiles(dt)
        
        # === Fire New Projectiles ===
        if fire_mask is not None:
            fire_mask = fire_mask.to(device=self.device, dtype=torch.bool)
            self.fire_projectiles_batched(fire_mask)
        
        # === Update Metrics ===
        self.total_time = self.total_time + dt
        
        # === Compute Ideal Lead Angles ===
        self._compute_ideal_lead(dt)
        
        # === Check Terminal Conditions ===
        robot_out_of_bounds = (
            (self.robot_x < 0) | (self.robot_x > WIDTH) |
            (self.robot_y < 0) | (self.robot_y > HEIGHT)
        )
        
        return {
            "new_hits": new_hits,
            "robot_out_of_bounds": robot_out_of_bounds,
        }

    def step_with_model_output(
        self,
        dt: torch.Tensor,
        target_yaw: torch.Tensor,
        target_pitch: torch.Tensor,
        time_to_fire: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method matching original API.
        Converts absolute targets to deltas and handles firing.
        
        Args:
            dt: (N,) or scalar
            target_yaw: (N,) desired camera yaw
            target_pitch: (N,) desired camera pitch
            time_to_fire: (N,) time until firing (fires when <= 0)
            
        Returns:
            info dict from step()
        """
        # Convert to deltas
        d_theta = torch.atan2(
            torch.sin(target_yaw - self.camera_theta),
            torch.cos(target_yaw - self.camera_theta)
        )
        d_pitch = target_pitch - self.camera_pitch
        
        # Determine which envs should fire
        fire_mask = time_to_fire <= 0.0
        
        return self.step(dt, d_theta, d_pitch, fire_mask)

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return all state tensors as a dictionary (for checkpointing)."""
        return {
            # Camera
            "camera_x": self.camera_x,
            "camera_y": self.camera_y,
            "camera_z": self.camera_z,
            "camera_theta": self.camera_theta,
            "camera_pitch": self.camera_pitch,
            # Robot
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "robot_z": self.robot_z,
            "robot_vx": self.robot_vx,
            "robot_vy": self.robot_vy,
            "robot_theta": self.robot_theta,
            "robot_omega": self.robot_omega,
            # Panels
            "panel_x": self.panel_x,
            "panel_y": self.panel_y,
            "panel_z": self.panel_z,
            "panel_visible": self.panel_visible,
            # Projectiles
            "proj_alive": self.proj_alive,
            # Metrics
            "hit_count": self.hit_count,
            "shots_fired": self.shots_fired,
            "total_time": self.total_time,
            # Lead
            "ideal_yaw": self.ideal_yaw,
            "ideal_pitch": self.ideal_pitch,
        }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_simulator(
    n_envs: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> VectorizedSimulator:
    """
    Factory function to create a vectorized simulator.
    
    Args:
        n_envs: Number of parallel environments
        device: PyTorch device (default: cuda if available, else cpu)
        dtype: Data type for tensors
        
    Returns:
        VectorizedSimulator instance with all envs reset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sim = VectorizedSimulator(n_envs=n_envs, device=device, dtype=dtype)
    sim.reset()
    
    return sim


@torch.jit.script
def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


@torch.jit.script
def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute smallest signed angle difference from b to a."""
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))