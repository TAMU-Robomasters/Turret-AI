import numpy as np
import cv2
import random
import time

# Coordinate system and units:
# - All distances are in millimeters (mm)
# - Velocities are in mm/s
# - Accelerations are in mm/s^2
# - Time is in seconds

WIDTH = 6000          # mm
HEIGHT = 6000         # mm
DEBUG = True

GRAVITY = -9800.0     # mm/s^2, downward acceleration

MAX_VEL = 4000        # mm/s


class Panel:
    def __init__(self, z_offset, base_theta):
        self.base_theta = base_theta
        self.z_offset = z_offset
        self.x = 0
        self.y = 0
        self.z = 0
        self.visible = False


class Camera:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.z = 0
        self.theta = np.deg2rad(-90)  # initial orientation
        self.pitch = 0.0  # 0 = level, negative = down
        self.fov = np.deg2rad(60)


class Robot:
    def __init__(self, radius, camera):
        # Randomize start away from camera
        self.radius = radius
        self.x, self.y = self._random_start_position(camera, min_dist=300)
        self.z = 0

        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.theta = 0.0
        self.omega = 0.0

        # Panels are offset vertically from the robot center (mm)
        height1 = random.uniform(-30.0, -10.0)
        height2 = random.uniform(-30.0, -10.0)

        self.panels = [
            Panel(height1, np.deg2rad(0)),
            Panel(height2, np.deg2rad(180)),
            Panel(height1, np.deg2rad(90)),
            Panel(height2, np.deg2rad(270)),
        ]

        self.update_panels(camera)

    def _random_start_position(self, camera, min_dist=300.0):
        while True:
            x = random.uniform(self.radius, WIDTH - self.radius)
            y = random.uniform(self.radius, HEIGHT - self.radius)
            dx = x - camera.x
            dy = y - camera.y
            if np.hypot(dx, dy) >= min_dist:
                return x, y

    def update_panels(self, camera):
        for panel in self.panels:
            theta = self.theta + panel.base_theta
            panel.x = self.x + self.radius * np.cos(theta)
            panel.y = self.y + self.radius * np.sin(theta)
            panel.z = self.z + panel.z_offset

            # Vector from camera to panel
            dx = panel.x - camera.x
            dy = panel.y - camera.y
            angle_to_panel = np.arctan2(dy, dx)
            relative_angle = angle_to_panel - camera.theta
            relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

            in_fov = abs(relative_angle) <= camera.fov / 2

            # Panel outward normal
            nx = np.cos(theta)
            ny = np.sin(theta)

            # Direction from panel to camera
            vx = camera.x - panel.x
            vy = camera.y - panel.y
            facing_camera = (nx * vx + ny * vy) > 0

            panel.visible = in_fov and facing_camera

    def update(self, dt, camera):
        # Occasional maneuver
        if random.random() < 0.05:
            self.ax = random.uniform(-750.0, 750.0)
            self.ay = random.uniform(-750.0, 750.0)
            self.omega = random.uniform(-6.0, 6.0)

        # Velocity update
        self.vx += self.ax * dt
        self.vy += self.ay * dt

        # Limit velocities
        self.vx = np.clip(self.vx, -MAX_VEL, MAX_VEL)
        self.vy = np.clip(self.vy, -MAX_VEL, MAX_VEL)

        # Position update
        self.x += self.vx * dt + 0.5 * self.ax * dt * dt
        self.y += self.vy * dt + 0.5 * self.ay * dt * dt

        # Rotation
        self.theta += self.omega * dt

        self.update_panels(camera)
        self.avoid_camera(camera, min_dist=300)  # Prevent moving into camera

    def avoid_camera(self, camera, min_dist=300.0):
        dx = self.x - camera.x
        dy = self.y - camera.y
        dist = np.hypot(dx, dy)

        if dist < min_dist:
            # Push robot outside the minimum distance
            angle = np.arctan2(dy, dx)
            self.x = camera.x + min_dist * np.cos(angle)
            self.y = camera.y + min_dist * np.sin(angle)

            # Remove velocity towards the camera
            v_radial = self.vx * np.cos(angle) + self.vy * np.sin(angle)
            if v_radial < 0:
                self.vx -= v_radial * np.cos(angle)
                self.vy -= v_radial * np.sin(angle)

class Projectile:
    def __init__(self, x, y, z, theta, pitch, speed):
        self.x = x
        self.y = y
        self.z = z
        # Decompose speed into horizontal and vertical using pitch
        horiz_speed = speed * np.cos(pitch)
        self.vx = horiz_speed * np.cos(theta)
        self.vy = horiz_speed * np.sin(theta)
        self.vz = speed * np.sin(pitch)
        self.alive = True

    def update(self, dt):
        if not self.alive:
            return

        # Integrate velocity with gravity in z
        self.vz += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Kill projectile if it leaves the world bounds or falls too far below
        if (
            self.x < 0
            or self.x > WIDTH
            or self.y < 0
            or self.y > HEIGHT
            or self.z < -20000
        ):
            self.alive = False


class Simulator:
    def __init__(self):
        self.camera = Camera()
        radius = random.randint(150, 250)
        self.robot = Robot(radius, self.camera)
        self.projectiles = []
        self.projectile_speed = 23000.0  # mm/s

        # Scoring / metrics
        self.total_time = 0.0
        self.opportunity_time = 0.0  # time robot is within FOV (current center)
        self.ideal_alignment_time = 0.0  # time camera is near ideal lead angle
        self.hit_count = 0
        self.shots_fired = 0  # total number of projectiles spawned

        # For estimating relative velocity of the robot center (camera-centered)
        self._prev_rel_center = None

        # Store latest ideal lead direction for debugging visualization
        self.ideal_yaw = None
        self.ideal_pitch = None

        # Aim camera so that the robot starts somewhere inside the FOV cone,
        # but not perfectly centered every time.
        dx = self.robot.x - self.camera.x
        dy = self.robot.y - self.camera.y
        base_yaw = np.arctan2(dy, dx)
        # Random offset so robot is somewhere within the cone, not dead center
        yaw_offset = random.uniform(-self.camera.fov / 4, self.camera.fov / 4)
        self.camera.theta = base_yaw + yaw_offset

    def _get_target_panel(self):
        """
        Select a single target panel for the model to track.
        Preference is given to the closest visible panel; if none
        are visible, fall back to the closest panel in 3D distance.
        """
        closest_visible = None
        closest_visible_dist = float("inf")
        closest_any = None
        closest_any_dist = float("inf")

        for panel in self.robot.panels:
            dx = panel.x - self.camera.x
            dy = panel.y - self.camera.y
            dz = panel.z - self.camera.z
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < closest_any_dist:
                closest_any_dist = dist
                closest_any = panel

            if panel.visible and dist < closest_visible_dist:
                closest_visible_dist = dist
                closest_visible = panel

        return closest_visible if closest_visible is not None else closest_any

    def fire_projectile(self):
        # Spawn a projectile from the camera along its current heading
        proj = Projectile(
            self.camera.x,
            self.camera.y,
            self.camera.z,
            self.camera.theta,
            self.camera.pitch,
            self.projectile_speed,
        )
        self.projectiles.append(proj)
        self.shots_fired += 1

    def _update_projectiles(self, dt, hit_tol_mm=100.0):
        alive_projectiles = []
        for proj in self.projectiles:
            proj.update(dt)
            if not proj.alive:
                continue

            hit = False
            for panel in self.robot.panels:
                # 3D positional tolerance check (sphere of radius hit_tol_mm)
                dx = proj.x - panel.x
                dy = proj.y - panel.y
                dz = proj.z - panel.z
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                if dist <= hit_tol_mm:
                    hit = True
                    if DEBUG:
                        print("hit")
                    self.hit_count += 1
                    break

            if not hit:
                alive_projectiles.append(proj)

        self.projectiles = alive_projectiles

    def step(self, dt, d_theta=0.0, d_pitch=0.0):
        # Camera control is now owned by the agent (RL policy).
        # d_theta, d_pitch are incremental changes per step.
        self.camera.theta += d_theta
        self.camera.pitch += d_pitch

        # Optional: clamp pitch to a reasonable range (e.g., ±90 degrees)
        max_pitch = np.deg2rad(89)
        self.camera.pitch = np.clip(self.camera.pitch, -max_pitch, max_pitch)

        # Update robot state
        self.robot.update(dt, self.camera)
        self._update_projectiles(dt)

        # Update scoring metrics
        # Camera-centered robot center
        rel_x = self.robot.x - self.camera.x
        rel_y = self.robot.y - self.camera.y
        rel_z = self.robot.z - self.camera.z

        # Estimate relative velocity via finite differences
        if self._prev_rel_center is not None and dt > 0:
            vx = (rel_x - self._prev_rel_center[0]) / dt
            vy = (rel_y - self._prev_rel_center[1]) / dt
            vz = (rel_z - self._prev_rel_center[2]) / dt
        else:
            vx = vy = vz = 0.0
        self._prev_rel_center = (rel_x, rel_y, rel_z)

        # Track total simulated time
        self.total_time += dt

        # "Opportunity" time: robot center currently within camera FOV in yaw
        center_yaw = np.arctan2(rel_y, rel_x)
        yaw_err_current = np.arctan2(
            np.sin(center_yaw - self.camera.theta),
            np.cos(center_yaw - self.camera.theta),
        )
        if abs(yaw_err_current) <= self.camera.fov / 2:
            self.opportunity_time += dt

        # Ideal lead direction toward future robot center based on simple intercept
        # approximation (ignores gravity for this alignment metric).
        p = np.array([rel_x, rel_y, rel_z], dtype=float)
        v = np.array([vx, vy, vz], dtype=float)
        p_norm = np.linalg.norm(p)
        if p_norm > 1e-6 and self.projectile_speed > 1e-6:
            T = p_norm / self.projectile_speed
            # A couple of fixed-point refinement steps for T
            for _ in range(2):
                future_pos = p + v * T
                dist_future = np.linalg.norm(future_pos)
                if dist_future < 1e-6:
                    break
                T = dist_future / self.projectile_speed

            future_pos = p + v * T
            fx, fy, fz = future_pos
            horiz_dist = np.hypot(fx, fy)
            ideal_yaw = np.arctan2(fy, fx)
            ideal_pitch = np.arctan2(fz, horiz_dist)

            # Store for debug drawing
            self.ideal_yaw = ideal_yaw
            self.ideal_pitch = ideal_pitch

            # Angular error between camera orientation and ideal lead angle
            yaw_err_ideal = np.arctan2(
                np.sin(ideal_yaw - self.camera.theta),
                np.cos(ideal_yaw - self.camera.theta),
            )
            pitch_err_ideal = ideal_pitch - self.camera.pitch

            # Count time when camera is well aligned with the ideal lead direction
            yaw_thresh = np.deg2rad(3.0)
            pitch_thresh = np.deg2rad(3.0)
            if abs(yaw_err_ideal) <= yaw_thresh and abs(pitch_err_ideal) <= pitch_thresh:
                self.ideal_alignment_time += dt

        # Default return (can be replaced by higher-level RL env wrapper)
        return None

    def get_model_input(self):
        """
        Return a compact feature vector for the model based on the
        current state and a single target panel.

        Features (all scalars):
        - camera_yaw
        - camera_pitch
        - yaw_to_panel        (global yaw from camera to panel)
        - pitch_to_panel      (global pitch from camera to panel)
        - panel_yaw_rel       (panel outward normal yaw relative to camera yaw)
        - distance_to_panel   (3D distance, mm)
        - projectile_speed    (mm/s)
        """
        target = self._get_target_panel()
        if target is None:
            # No panels (should not happen), return zeros
            return np.zeros(7, dtype=float)

        # Relative position from camera to panel (mm)
        rel_x = target.x - self.camera.x
        rel_y = target.y - self.camera.y
        rel_z = target.z - self.camera.z

        horiz_dist = np.hypot(rel_x, rel_y)
        distance = np.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)

        # Global yaw/pitch from camera to panel
        yaw_to_panel = np.arctan2(rel_y, rel_x)
        pitch_to_panel = np.arctan2(rel_z, horiz_dist) if horiz_dist > 1e-6 else 0.0

        # Panel outward normal yaw (world), then relative to camera yaw
        # Recompute panel orientation the same way as in update_panels.
        theta_panel = self.robot.theta + target.base_theta
        panel_yaw_world = theta_panel
        panel_yaw_rel = np.arctan2(
            np.sin(panel_yaw_world - self.camera.theta),
            np.cos(panel_yaw_world - self.camera.theta),
        )

        features = np.array(
            [
                self.camera.theta,
                self.camera.pitch,
                yaw_to_panel,
                pitch_to_panel,
                panel_yaw_rel,
                distance,
                self.projectile_speed,
            ],
            dtype=float,
        )
        return features

    def step_with_model_output(self, dt, target_yaw, target_pitch, time_to_fire):
        """
        Convenience helper for RL / RNN inference.

        The model outputs:
        - target_yaw: desired camera yaw (rad, world frame)
        - target_pitch: desired camera pitch (rad)
        - time_to_fire: predicted time until firing (s, relative to *now*)

        This helper:
        - Converts (target_yaw, target_pitch) into incremental deltas
          for this step and advances the simulation.
        - If time_to_fire <= 0, it also fires a projectile.

        Note: managing the countdown of time_to_fire across steps
        (e.g., subtracting dt each loop) is the caller's responsibility.
        """
        # Smallest-angle difference for yaw
        d_theta = np.arctan2(
            np.sin(target_yaw - self.camera.theta),
            np.cos(target_yaw - self.camera.theta),
        )
        d_pitch = target_pitch - self.camera.pitch

        self.step(dt, d_theta=d_theta, d_pitch=d_pitch)

        if time_to_fire <= 0.0:
            self.fire_projectile()

    def observe(self, noise=True):
        obs = []
        for panel in self.robot.panels:
            # Express panel positions in a camera-centered frame so that
            # the origin (0, 0, 0) is at the camera position.
            x = panel.x - self.camera.x
            y = panel.y - self.camera.y
            z = panel.z - self.camera.z
            if noise:
                x += np.random.normal(0, 2)
                y += np.random.normal(0, 2)
                z += np.random.normal(0, 1)
            obs.append([x, y, z])
        return np.array(obs)

    def render(self):
        """
        Debug rendering using OpenCV. This visualizes the current
        simulator state (camera, robot, panels, projectiles and
        ideal lead direction).
        """
        if not DEBUG:
            return

        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Draw robot
        cv2.circle(canvas, (int(self.robot.x), int(self.robot.y)), self.robot.radius, (255, 255, 255), 1)

        # Draw camera
        cv2.circle(canvas, (int(self.camera.x), int(self.camera.y)), 100, (0, 255, 0), -1)

        # Draw FOV
        fov_left_angle = self.camera.theta - self.camera.fov / 2
        fov_right_angle = self.camera.theta + self.camera.fov / 2
        fov_left_x = int(self.camera.x + 1000 * np.cos(fov_left_angle))
        fov_left_y = int(self.camera.y + 1000 * np.sin(fov_left_angle))
        fov_right_x = int(self.camera.x + 1000 * np.cos(fov_right_angle))
        fov_right_y = int(self.camera.y + 1000 * np.sin(fov_right_angle))

        cv2.line(canvas, (int(self.camera.x), int(self.camera.y)), (fov_left_x, fov_left_y), (0, 255, 255), 1)
        cv2.line(canvas, (int(self.camera.x), int(self.camera.y)), (fov_right_x, fov_right_y), (0, 255, 255), 1)

        # Draw panels
        for panel in self.robot.panels:
            if panel.visible:
                cv2.circle(canvas, (int(panel.x), int(panel.y)), 50, (255, 0, 0), -1)

        # Draw line to closest visible panel
        closest_panel = None
        closest_dist = float("inf")
        for panel in self.robot.panels:
            if panel.visible:
                dx = panel.x - self.camera.x
                dy = panel.y - self.camera.y
                dist = np.hypot(dx, dy)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_panel = panel
        if closest_panel:
            cv2.line(
                canvas,
                (int(self.camera.x), int(self.camera.y)),
                (int(closest_panel.x), int(closest_panel.y)),
                (0, 0, 255),
                2,
            )

        # Draw ideal lead direction (purple) if available
        if self.ideal_yaw is not None:
            line_len = 1000
            ix = int(self.camera.x + line_len * np.cos(self.ideal_yaw))
            iy = int(self.camera.y + line_len * np.sin(self.ideal_yaw))
            cv2.line(
                canvas,
                (int(self.camera.x), int(self.camera.y)),
                (ix, iy),
                (255, 0, 255),
                2,
            )

        # Draw projectiles
        for proj in self.projectiles:
            if proj.alive:
                cv2.circle(canvas, (int(proj.x), int(proj.y)), 20, (0, 255, 255), -1)

        debug_view = cv2.resize(canvas, (1000, 1000))
        cv2.imshow("sim", debug_view)
        # Small delay to update window; do not block on key here.
        cv2.waitKey(1)


if __name__ == "__main__":
    sim = Simulator()

    while True:
        start_time = time.perf_counter()
        dt = random.uniform(.012, .04)
        # For now, no external agent hooked up here; pass zero deltas.
        sim.step(dt, d_theta=0.0, d_pitch=0.0)
        robot = sim.robot

        if DEBUG:
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # Draw robot
            cv2.circle(canvas, (int(robot.x), int(robot.y)), robot.radius, (255, 255, 255), 1)

            # Draw camera
            cv2.circle(canvas, (int(sim.camera.x), int(sim.camera.y)), 100, (0, 255, 0), -1)

            # Draw FOV
            fov_left_angle = sim.camera.theta - sim.camera.fov / 2
            fov_right_angle = sim.camera.theta + sim.camera.fov / 2
            fov_left_x = int(sim.camera.x + 1000 * np.cos(fov_left_angle))
            fov_left_y = int(sim.camera.y + 1000 * np.sin(fov_left_angle))
            fov_right_x = int(sim.camera.x + 1000 * np.cos(fov_right_angle))
            fov_right_y = int(sim.camera.y + 1000 * np.sin(fov_right_angle))

            cv2.line(canvas, (int(sim.camera.x), int(sim.camera.y)), (fov_left_x, fov_left_y), (0, 255, 255), 1)
            cv2.line(canvas, (int(sim.camera.x), int(sim.camera.y)), (fov_right_x, fov_right_y), (0, 255, 255), 1)

            # Draw panels
            for panel in robot.panels:
                if panel.visible:
                    cv2.circle(canvas, (int(panel.x), int(panel.y)), 50, (255, 0, 0), -1)

            # Draw line to closest visible panel
            closest_panel = None
            closest_dist = float("inf")
            for panel in robot.panels:
                if panel.visible:
                    dx = panel.x - sim.camera.x
                    dy = panel.y - sim.camera.y
                    dist = np.hypot(dx, dy)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_panel = panel
            if closest_panel:
                cv2.line(canvas,
                         (int(sim.camera.x), int(sim.camera.y)),
                         (int(closest_panel.x), int(closest_panel.y)),
                         (0, 0, 255), 2)

            # Draw ideal lead direction (purple) if available
            if sim.ideal_yaw is not None:
                line_len = 1000
                ix = int(sim.camera.x + line_len * np.cos(sim.ideal_yaw))
                iy = int(sim.camera.y + line_len * np.sin(sim.ideal_yaw))
                cv2.line(canvas,
                         (int(sim.camera.x), int(sim.camera.y)),
                         (ix, iy),
                         (255, 0, 255), 2)

            # Draw projectiles
            for proj in sim.projectiles:
                if proj.alive:
                    cv2.circle(canvas, (int(proj.x), int(proj.y)), 20, (0, 255, 255), -1)

            debug_view = cv2.resize(canvas, (1000, 1000))
            cv2.imshow("sim", debug_view)
            key = cv2.waitKey(10)
            if key == 27:
                break

        # Stop simulation if robot leaves bounds
        if robot.x < 0 or robot.x > WIDTH or robot.y < 0 or robot.y > HEIGHT:
            break

        end = time.perf_counter()