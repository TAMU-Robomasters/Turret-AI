import numpy as np
import random

from turret_sim import Simulator, WIDTH, HEIGHT


class TurretEnv:
    """
    Minimal RL-style environment wrapper around Simulator.

    API:
      - obs = env.reset()
      - obs, reward, done, info = env.step(action)

    Where:
      - obs: normalized feature vector from Simulator.get_model_input()
      - action: np.array([target_yaw, target_pitch, time_to_fire])
                (radians, radians, seconds)
    """

    def __init__(
        self,
        dt: float = 0.015,
        max_steps: int = 500,
        obs_history_k: int = 4,
        action_is_delta: bool = False,
        action_is_correction: bool = False,
        correction_baseline: str = "panel",
        correction_clip_yaw: float | None = float(np.deg2rad(30.0)),
        correction_clip_pitch: float | None = float(np.deg2rad(15.0)),
        maintain_fire_timer: bool = True,
        lost_done_steps: int = 150,
        lost_penalty_base: float = -200.0,
        lost_penalty_slope: float = -20.0,
        lost_penalty_cap_steps: int = 50,
        lost_terminal_penalty: float = -5000.0,
        shot_base_penalty: float = 10.0,
        shot_miss_penalty_scale: float = 20.0,
        shot_miss_target_radius_mm: float = 60.0,
        shot_miss_penalty_power: float = 1.0,
        shot_miss_max_ratio: float = 5.0,
        shot_miss_max_angle_deg: float = 45.0,
        # Alignment reward shaping (camera following ideal lead angle).
        # - Bonus term uses a linear "align score" that drops to 0 beyond `align_angle_scale_deg`.
        # - Punishment term uses a smooth quadratic/quartic penalty on angular error.
        align_bonus_scale: float = 25.0,
        align_penalty_coeff: float = 600.0,
        # Lower power makes small angle errors hurt more immediately.
        align_penalty_power: float = 1.0,
        align_angle_scale_deg: float = 45.0,
        seed: int | None = None,
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.action_is_delta = action_is_delta
        self.action_is_correction = action_is_correction
        self.correction_baseline = correction_baseline
        self.correction_clip_yaw = None if correction_clip_yaw is None else float(correction_clip_yaw)
        self.correction_clip_pitch = None if correction_clip_pitch is None else float(correction_clip_pitch)
        self.maintain_fire_timer = maintain_fire_timer
        self.lost_done_steps = int(lost_done_steps)
        self.lost_penalty_base = float(lost_penalty_base)
        self.lost_penalty_slope = float(lost_penalty_slope)
        self.lost_penalty_cap_steps = int(lost_penalty_cap_steps)
        self.lost_terminal_penalty = float(lost_terminal_penalty)
        self.shot_base_penalty = float(shot_base_penalty)
        self.shot_miss_penalty_scale = float(shot_miss_penalty_scale)
        self.shot_miss_target_radius_mm = float(shot_miss_target_radius_mm)
        self.shot_miss_penalty_power = float(shot_miss_penalty_power)
        self.shot_miss_max_ratio = float(shot_miss_max_ratio)
        self.shot_miss_max_angle_deg = float(shot_miss_max_angle_deg)
        self.align_bonus_scale = float(align_bonus_scale)
        self.align_penalty_coeff = float(align_penalty_coeff)
        self.align_penalty_power = float(align_penalty_power)
        self.align_angle_scale_deg = float(align_angle_scale_deg)
        self.seed = None if seed is None else int(seed)

        # Scaling constants for normalization
        self.max_distance = float(np.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT))
        self.speed_scale = 23000.0  # mm/s, rough scale for projectile speed
        self.base_obs_dim = 4
        self.obs_history_k = int(obs_history_k)
        if self.obs_history_k < 1:
            raise ValueError(f"obs_history_k must be >= 1, got {self.obs_history_k}")
        self.obs_dim = self.base_obs_dim * self.obs_history_k

        self.sim: Simulator | None = None
        self.steps = 0
        self.prev_hit_count = 0
        self.prev_shots_fired = 0
        self.lost_steps = 0  # how many consecutive steps target has been off-screen
        self._time_to_fire_remaining: float | None = None
        self._prev_d_theta = 0.0
        self._prev_d_pitch = 0.0
        self._obs_history: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the simulation and return the initial observation."""
        if seed is None:
            seed = self.seed
        if seed is not None:
            # Simulator uses both Python's `random` and NumPy RNG.
            random.seed(int(seed))
            np.random.seed(int(seed))
        self.sim = Simulator()
        self.steps = 0
        self.prev_hit_count = 0
        self.prev_shots_fired = 0
        self.lost_steps = 0
        self._time_to_fire_remaining = None
        self._prev_d_theta = 0.0
        self._prev_d_pitch = 0.0
        self.last_reward_parts = {
            "hit": 0.0,
            "shot": 0.0,
            "align": 0.0,
            "track": 0.0,
        }
        obs_base = self._get_obs_base()
        self._reset_history(obs_base)
        return self._get_obs_from_history()

    def step(self, action: np.ndarray):
        """
        Advance the environment by one step using the given action.

        action: np.array([a0, a1, time_to_fire_cmd])
          - Default (action_is_delta=False and action_is_correction=False):
              a0/a1 are absolute targets:
                - a0: desired camera yaw (radians, world frame)
                - a1: desired camera pitch (radians)
          - If action_is_delta=True:
              a0/a1 are incremental offsets applied to the current camera pose:
                - target_yaw = camera_yaw + a0
                - target_pitch = camera_pitch + a1
          - If action_is_correction=True:
              a0/a1 are corrections applied to a per-step non-ML baseline:
                - baseline is recomputed every step
                - correction_baseline="panel" uses yaw/pitch-to-target-panel
                - target_yaw = baseline_yaw + a0
                - target_pitch = baseline_pitch + a1
          - time_to_fire_cmd: time until firing (seconds).
              This is a continuous *time-to-fire in seconds* (not a probability).
              - If maintain_fire_timer=True (default): this value arms the
                robot's internal countdown timer (if not already armed).
                The timer then counts down by dt each step and fires when <= 0.
              - If maintain_fire_timer=False: this value is treated as
                "time until firing relative to now" for *this step only*;
                when <= dt a projectile will be fired this step.
        """
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        if action.shape[0] not in (2, 3):
            raise ValueError("Action must have shape (2,) or (3,), got %s" % (action.shape,))

        a0_raw = float(action[0])
        if action.shape[0] == 3:
            a1_raw = float(action[1])
            time_to_fire_cmd = float(action[2])
        else:
            a1_raw = 0.0
            time_to_fire_cmd = float(action[1])
        a0 = a0_raw
        a1 = a1_raw
        if self.action_is_delta and self.action_is_correction:
            raise ValueError("Choose only one of action_is_delta or action_is_correction.")

        baseline_yaw = None
        baseline_pitch = None
        if self.action_is_correction:
            if self.correction_clip_yaw is not None:
                a0 = float(np.clip(a0, -self.correction_clip_yaw, self.correction_clip_yaw))
            if self.correction_clip_pitch is not None:
                a1 = float(np.clip(a1, -self.correction_clip_pitch, self.correction_clip_pitch))
            if self.correction_baseline == "panel":
                raw = self.sim.get_model_input()
                baseline_yaw = float(raw[2])
                baseline_pitch = float(raw[3])
            elif self.correction_baseline == "ideal":
                baseline_yaw = float(self.sim.ideal_yaw) if self.sim.ideal_yaw is not None else 0.0
                baseline_pitch = float(self.sim.ideal_pitch) if self.sim.ideal_pitch is not None else 0.0
            else:
                raise ValueError(
                    "Unknown correction_baseline=%r (expected 'panel' or 'ideal')." % (self.correction_baseline,)
                )
            target_yaw = float(baseline_yaw + a0)
            target_pitch = float(baseline_pitch)
        elif self.action_is_delta:
            target_yaw = float(self.sim.camera.theta + a0)
            target_pitch = float(self.sim.camera.pitch + a1)
        else:
            target_yaw = a0
            target_pitch = a1

        # Match real robot behavior: maintain a countdown timer internally.
        # This allows the policy to set a time-to-fire once and have it count down.
        if self.maintain_fire_timer:
            if self._time_to_fire_remaining is None:
                self._time_to_fire_remaining = time_to_fire_cmd

            # Countdown occurs in real time regardless of action updates.
            self._time_to_fire_remaining -= self.dt
            time_to_fire = self._time_to_fire_remaining
        else:
            time_to_fire = time_to_fire_cmd - self.dt

        # Advance the simulator with model outputs
        shots_before = self.sim.shots_fired
        self.sim.step_with_model_output(self.dt, target_yaw, target_pitch, time_to_fire)
        if self.maintain_fire_timer and self.sim.shots_fired > shots_before:
            # Reset so the next shot requires re-arming the timer.
            self._time_to_fire_remaining = None
        self.steps += 1

        obs = self._get_obs()
        reward, reward_parts = self._compute_reward()
        self.last_reward_parts = reward_parts
        done = self._is_done()
        info = {
            "hit_count": self.sim.hit_count,
            "total_time": self.sim.total_time,
            "opportunity_time": self.sim.opportunity_time,
            "ideal_alignment_time": self.sim.ideal_alignment_time,
            "target_yaw": float(target_yaw),
            "target_pitch": float(target_pitch),
            "last_shot_visible_panel_count": int(getattr(self.sim, "last_shot_visible_panel_count", 0) or 0),
            "reward_parts": reward_parts,
        }
        if self.action_is_correction:
            info.update(
                {
                    "baseline_yaw": float(baseline_yaw) if baseline_yaw is not None else None,
                    "baseline_pitch": float(baseline_pitch) if baseline_pitch is not None else None,
                    "correction_yaw_raw": float(a0_raw),
                    "correction_pitch_raw": float(a1_raw),
                    "correction_yaw_applied": float(a0),
                    "correction_pitch_applied": float(a1),
                }
            )

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs_base(self) -> np.ndarray:
        """
        Fetch and normalize the base model input features from the simulator.

        Raw features from Simulator.get_model_input():
          [yaw_to_panel,
           panel_yaw_world,
           distance_to_panel_mm,
           projectile_speed_mm_per_s]

        Normalization:
          - All angles divided by pi  -> approx in [-1, 1]
          - Distance divided by max_distance
          - Speed divided by speed_scale
        """
        raw = self.sim.get_model_input()

        yaw_to_panel = raw[0] / np.pi
        panel_yaw_world = raw[1] / np.pi

        distance = raw[2] / max(self.max_distance, 1e-6)
        projectile_speed = raw[3] / self.speed_scale

        return np.array(
            [
                yaw_to_panel,
                panel_yaw_world,
                distance,
                projectile_speed,
            ],
            dtype=np.float32,
        )

    def _reset_history(self, obs_base: np.ndarray) -> None:
        obs_base = np.asarray(obs_base, dtype=np.float32)
        obs_repeat = np.repeat(obs_base[None, :], self.obs_history_k, axis=0)
        self._obs_history = obs_repeat

    def _update_history(self, obs_base: np.ndarray) -> None:
        obs_base = np.asarray(obs_base, dtype=np.float32)
        if self._obs_history is None or self._obs_history.shape != (self.obs_history_k, self.base_obs_dim):
            self._reset_history(obs_base)
            return
        if self.obs_history_k == 1:
            self._obs_history[0] = obs_base
            return
        self._obs_history[:-1] = self._obs_history[1:]
        self._obs_history[-1] = obs_base

    def _get_obs_from_history(self) -> np.ndarray:
        if self._obs_history is None:
            return self._get_obs_base()
        return self._obs_history.reshape(-1).astype(np.float32, copy=False)

    def _get_obs(self) -> np.ndarray:
        obs_base = self._get_obs_base()
        self._update_history(obs_base)
        return self._get_obs_from_history()

    def _compute_reward(self):
        """
        Combine hit-based reward with alignment-based shaping.
        """
        # Hit reward: positive reward when new hits occur
        hit_inc = self.sim.hit_count - self.prev_hit_count
        self.prev_hit_count = self.sim.hit_count
        r_hit = 110.0 * hit_inc

        # Shot penalty: small negative reward for each projectile fired
        shots_inc = self.sim.shots_fired - self.prev_shots_fired
        self.prev_shots_fired = self.sim.shots_fired
        # Penalize firing, and add an extra miss-distance penalty based on aim error
        # at the instant the shot is fired (dense learning signal).
        r_shot_base = -self.shot_base_penalty * shots_inc
        r_shot_miss = 0.0
        r_blind_fire = 0.0
        if shots_inc > 0:
            visible_cnt = int(getattr(self.sim, "last_shot_visible_panel_count", 0) or 0)
            if visible_cnt <= 0:
                # In the real system you shouldn't be able to score hits on an unseen panel.
                # Strongly discourage firing when no valid target is visible.
                r_blind_fire = -500.0 * shots_inc
            elif self.shot_miss_penalty_scale != 0.0:
                ideal_yaw = getattr(self.sim, "ideal_yaw", None)
                ideal_pitch = getattr(self.sim, "ideal_pitch", None)
                if ideal_yaw is not None and ideal_pitch is not None:
                    yaw_err = np.arctan2(
                        np.sin(float(ideal_yaw) - float(self.sim.camera.theta)),
                        np.cos(float(ideal_yaw) - float(self.sim.camera.theta)),
                    )
                    pitch_err = float(ideal_pitch) - float(self.sim.camera.pitch)
                    angle_err = float(np.sqrt(yaw_err * yaw_err + pitch_err * pitch_err))
                    max_angle_rad = float(np.deg2rad(max(1e-6, self.shot_miss_max_angle_deg)))
                    angle_err = float(np.clip(angle_err, 0.0, max_angle_rad))

                    raw = self.sim.get_model_input()
                    try:
                        distance_mm = float(raw[5])
                    except Exception:
                        distance_mm = 0.0

                    # Approximate lateral miss distance from angular error.
                    miss_mm = float(distance_mm * np.tan(angle_err))
                    denom = max(float(self.shot_miss_target_radius_mm), 1e-6)
                    ratio = miss_mm / denom
                    ratio = float(np.clip(ratio, 0.0, max(0.0, self.shot_miss_max_ratio)))
                    power = max(float(self.shot_miss_penalty_power), 1e-6)
                    r_shot_miss = -self.shot_miss_penalty_scale * (ratio**power) * shots_inc

        # Alignment reward: encourage camera to track ideal lead angle
        r_align = 0.0
        if self.sim.ideal_yaw is not None:
            yaw_err = np.arctan2(
                np.sin(self.sim.ideal_yaw - self.sim.camera.theta),
                np.cos(self.sim.ideal_yaw - self.sim.camera.theta),
            )
            pitch_err = float(self.sim.ideal_pitch - self.sim.camera.pitch)
            angle_err = float(abs(yaw_err) + abs(pitch_err))

            # Following ideal/predicted line:
            # - bonus drops linearly with error up to `align_angle_scale_deg`
            # - penalty grows smoothly with a stronger power on error
            angle_scale = np.deg2rad(max(self.align_angle_scale_deg, 1e-6))
            align_score = max(0.0, 1.0 - angle_err / max(float(angle_scale), 1e-6))
            r_align = (
                self.align_bonus_scale * float(align_score)
                - self.align_penalty_coeff * (angle_err**self.align_penalty_power)
            )

        # Tracking reward: strongly encourage keeping the target panel in view.
        # If the chosen target panel is visible, reward staying locked on it.
        # If it is lost (off-screen), give a strong penalty and track how long
        # it has been lost.
        r_track = 0.0
        target_panel = self.sim._get_target_panel()
        if target_panel is not None and target_panel.visible:
            # Positive reward every step the target remains visible.
            r_track = 50.0
            self.lost_steps = 0
        else:
            self.lost_steps += 1
            # Softer (non-cliff) shaping: small penalty when first lost,
            # increasing with consecutive lost steps, plus a terminal penalty
            # when the episode ends due to being lost too long.
            capped = min(self.lost_steps, self.lost_penalty_cap_steps)
            r_track = self.lost_penalty_base + self.lost_penalty_slope * capped
            if self.lost_steps >= self.lost_done_steps:
                r_track += self.lost_terminal_penalty

        # Camera motion penalty: discourages rapid side-switching / jitter.
        d_theta = float(getattr(self.sim, "last_d_theta", 0.0))
        d_pitch = float(getattr(self.sim, "last_d_pitch", 0.0))
        r_motion = -50.0 * (abs(d_theta) + abs(d_pitch))

        jerk = abs(d_theta - self._prev_d_theta) + abs(d_pitch - self._prev_d_pitch)
        r_jerk = -200.0 * jerk
        self._prev_d_theta = d_theta
        self._prev_d_pitch = d_pitch

        r_shot = float(r_shot_base + r_shot_miss + r_blind_fire)
        reward_parts = {
            "hit": float(r_hit),
            "shot": r_shot,
            "align": float(r_align),
            "track": float(r_track),
        }
        total_reward = float(
            r_hit
            + r_shot
            + r_align
            + r_track
            + r_motion
            + r_jerk
        )
        return total_reward, reward_parts

    def _is_done(self) -> bool:
        """Episode termination condition."""
        if self.steps >= self.max_steps:
            return True

        # Terminate if robot leaves the world bounds
        robot = self.sim.robot
        if robot.x < 0 or robot.x > WIDTH or robot.y < 0 or robot.y > HEIGHT:
            return True

        # If the target has been off-screen for too long, end the episode.
        # This teaches the agent that losing the robot is a terminal failure.
        if self.lost_steps >= self.lost_done_steps:
            return True

        return False

    def render(self):
        """Render the underlying simulator debug view."""
        if self.sim is not None:
            self.sim.render()
