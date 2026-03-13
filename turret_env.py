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

    def __init__(self, dt: float = 0.015, max_steps: int = 500):
        self.dt = dt
        self.max_steps = max_steps

        # Scaling constants for normalization
        self.max_distance = float(np.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT))
        self.speed_scale = 3000.0  # mm/s, rough scale for projectile speed

        self.sim: Simulator | None = None
        self.steps = 0
        self.prev_hit_count = 0
        self.prev_shots_fired = 0
        self.lost_steps = 0  # how many consecutive steps target has been off-screen

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
        return self._get_obs()

    def step(self, action: np.ndarray):
        """
        Advance the environment by one step using the given action.

        action: np.array([target_yaw, target_pitch, time_to_fire])
          - target_yaw: desired camera yaw (radians, world frame)
          - target_pitch: desired camera pitch (radians)
          - time_to_fire: time until firing (seconds); when <= 0
                          a projectile will be fired this step.
        """
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        if action.shape[0] != 3:
            raise ValueError("Action must have shape (3,), got %s" % (action.shape,))

        target_yaw = float(action[0])
        target_pitch = float(action[1])
        # Keep the third component as an explicit "time to fire" in seconds,
        # matching the interface you will use on the real robot.
        time_to_fire = float(action[2])

        # Advance the simulator with model outputs
        self.sim.step_with_model_output(self.dt, target_yaw, target_pitch, time_to_fire)
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
           panel_yaw_rel,
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
        panel_yaw_rel = raw[4] / np.pi

        distance = raw[5] / max(self.max_distance, 1e-6)
        projectile_speed = raw[6] / self.speed_scale

        obs = np.array(
            [
                camera_yaw,
                camera_pitch,
                yaw_to_panel,
                pitch_to_panel,
                panel_yaw_rel,
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
        r_hit = 10.0 * hit_inc

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
            # Up to +2.0 for perfect alignment, minus a small penalty with distance
            r_align = 16.0 * align_score - 2.0 * angle_err

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
            # Penalize losing the target; stronger than shot penalty so the
            # agent prioritizes keeping the robot in view.
            r_track = -1000.0
            self.lost_steps += 1

        return float(r_hit + r_shot + r_align + r_track)

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
        if self.lost_steps >= 100:
            return True

        return False

    def render(self):
        """Render the underlying simulator debug view."""
        if self.sim is not None:
            self.sim.render()


