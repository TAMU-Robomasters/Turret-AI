import numpy as np

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
        action_is_delta: bool = False,
        action_is_correction: bool = False,
        correction_baseline: str = "panel",
        maintain_fire_timer: bool = True,
        lost_done_steps: int = 150,
        lost_penalty_base: float = -200.0,
        lost_penalty_slope: float = -20.0,
        lost_penalty_cap_steps: int = 50,
        lost_terminal_penalty: float = -5000.0,
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.action_is_delta = action_is_delta
        self.action_is_correction = action_is_correction
        self.correction_baseline = correction_baseline
        self.maintain_fire_timer = maintain_fire_timer
        self.lost_done_steps = int(lost_done_steps)
        self.lost_penalty_base = float(lost_penalty_base)
        self.lost_penalty_slope = float(lost_penalty_slope)
        self.lost_penalty_cap_steps = int(lost_penalty_cap_steps)
        self.lost_terminal_penalty = float(lost_terminal_penalty)

        # Scaling constants for normalization
        self.max_distance = float(np.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT))
        self.speed_scale = 23000.0  # mm/s, rough scale for projectile speed

        self.sim: Simulator | None = None
        self.steps = 0
        self.prev_hit_count = 0
        self.prev_shots_fired = 0
        self.lost_steps = 0  # how many consecutive steps target has been off-screen
        self._time_to_fire_remaining: float | None = None
        self._prev_d_theta = 0.0
        self._prev_d_pitch = 0.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset the simulation and return the initial observation."""
        self.sim = Simulator()
        self.steps = 0
        self.prev_hit_count = 0
        self.prev_shots_fired = 0
        self.lost_steps = 0
        self._time_to_fire_remaining = None
        self._prev_d_theta = 0.0
        self._prev_d_pitch = 0.0
        return self._get_obs()

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
              - If maintain_fire_timer=True (default): this value arms the
                robot's internal countdown timer (if not already armed).
                The timer then counts down by dt each step and fires when <= 0.
              - If maintain_fire_timer=False: this value is treated as
                "time until firing relative to now" for *this step only*;
                when <= 0 a projectile will be fired this step.
        """
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        if action.shape[0] != 3:
            raise ValueError("Action must have shape (3,), got %s" % (action.shape,))

        a0 = float(action[0])
        a1 = float(action[1])
        if self.action_is_delta and self.action_is_correction:
            raise ValueError("Choose only one of action_is_delta or action_is_correction.")

        if self.action_is_correction:
            if self.correction_baseline == "panel":
                raw = self.sim.get_model_input()
                baseline_yaw = float(raw[2])
                baseline_pitch = float(raw[3])
            else:
                raise ValueError(
                    "Unknown correction_baseline=%r (expected 'panel')." % (self.correction_baseline,)
                )
            target_yaw = float(baseline_yaw + a0)
            target_pitch = float(baseline_pitch + a1)
        elif self.action_is_delta:
            target_yaw = float(self.sim.camera.theta + a0)
            target_pitch = float(self.sim.camera.pitch + a1)
        else:
            target_yaw = a0
            target_pitch = a1
        time_to_fire_cmd = float(action[2])

        # Match real robot behavior: maintain a countdown timer internally.
        # This allows the policy to set a time-to-fire once and have it count down.
        if self.maintain_fire_timer:
            if self._time_to_fire_remaining is None:
                self._time_to_fire_remaining = time_to_fire_cmd

            # Countdown occurs in real time regardless of action updates.
            self._time_to_fire_remaining -= self.dt
            time_to_fire = self._time_to_fire_remaining
        else:
            time_to_fire = time_to_fire_cmd

        # Advance the simulator with model outputs
        shots_before = self.sim.shots_fired
        self.sim.step_with_model_output(self.dt, target_yaw, target_pitch, time_to_fire)
        if self.maintain_fire_timer and self.sim.shots_fired > shots_before:
            # Reset so the next shot requires re-arming the timer.
            self._time_to_fire_remaining = None
        self.steps += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {
            "hit_count": self.sim.hit_count,
            "total_time": self.sim.total_time,
            "opportunity_time": self.sim.opportunity_time,
            "ideal_alignment_time": self.sim.ideal_alignment_time,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """
        Fetch and normalize the model input features from the simulator.

        Raw features from Simulator.get_model_input():
          [camera_yaw,
           camera_pitch,
           yaw_to_panel,
           pitch_to_panel,
           panel_yaw_world,
           distance_to_panel_mm,
           projectile_speed_mm_per_s]

        Normalization:
          - All angles divided by pi  -> approx in [-1, 1]
          - Distance divided by max_distance
          - Speed divided by speed_scale
        """
        raw = self.sim.get_model_input()

        camera_yaw = raw[0] / np.pi
        camera_pitch = raw[1] / np.pi
        yaw_to_panel = raw[2] / np.pi
        pitch_to_panel = raw[3] / np.pi
        panel_yaw_world = raw[4] / np.pi

        distance = raw[5] / max(self.max_distance, 1e-6)
        projectile_speed = raw[6] / self.speed_scale

        obs = np.array(
            [
                camera_yaw,
                camera_pitch,
                yaw_to_panel,
                pitch_to_panel,
                panel_yaw_world,
                distance,
                projectile_speed,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self) -> float:
        """
        Combine hit-based reward with alignment-based shaping.
        """
        # Hit reward: positive reward when new hits occur
        hit_inc = self.sim.hit_count - self.prev_hit_count
        self.prev_hit_count = self.sim.hit_count
        r_hit = 100.0 * hit_inc

        # Shot penalty: small negative reward for each projectile fired
        shots_inc = self.sim.shots_fired - self.prev_shots_fired
        self.prev_shots_fired = self.sim.shots_fired
        # Stronger penalty per shot to discourage spamming
        r_shot = -15.0 * shots_inc

        # Alignment reward: encourage camera to track ideal lead angle
        r_align = 0.0
        if self.sim.ideal_yaw is not None:
            yaw_err = np.arctan2(
                np.sin(self.sim.ideal_yaw - self.sim.camera.theta),
                np.cos(self.sim.ideal_yaw - self.sim.camera.theta),
            )
            pitch_err = self.sim.ideal_pitch - self.sim.camera.pitch
            angle_err = abs(yaw_err) + abs(pitch_err)

            # Higher reward when closely following the ideal line:
            # map small angular error to a positive bonus, and keep
            # a mild penalty for being far away.
            angle_scale = np.deg2rad(45.0)  # beyond this, align bonus ~0
            align_score = max(0.0, 1.0 - angle_err / max(angle_scale, 1e-6))
            # Stronger punishment for being off the ideal line.
            # (angle_err is in radians; typical scale: 0.5 rad ~ 30 degrees)
            r_align = 10.0 * align_score - 120.0 * angle_err

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

        return float(r_hit + r_shot + r_align + r_track + r_motion + r_jerk)

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
