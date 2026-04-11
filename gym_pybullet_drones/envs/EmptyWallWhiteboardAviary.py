import pkg_resources
import time
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class EmptyWallWhiteboardAviary(BaseAviary):
    """Control-oriented env with only one wall and one whiteboard setup."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 240,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=True,
        vision_attributes=False,
        output_folder="results",
        ceiling_height: float = 4.0,
        wall_x_offset: float = 3.0,
    ):
        self.SCENE_CONFIG = {
            "seed": 0,
            "enable_whiteboard": True,
            "whiteboard_width": 1.84,
            "whiteboard_height": 1.20,
            "whiteboard_depth": 0.025,
            "whiteboard_bottom_z": 0.75,
            "pole_side": 0.05,
            "whiteboard_pos_xy": [0.0, 0.0],
            "whiteboard_yaw": 0.0,
            # Simple distractor obstacle (replaces old lightswitch).
            "enable_distractor": True,
            "distractor_pos_xy": [1.0, 1.0],
            "distractor_size": 0.25,
            "distractor_height_z": 1.0,
            "distractor_shape": "box",  # "box" or "cylinder"
            # Additional distractor scene elements.
            "enable_ceiling": False,
            "ceiling_height": 4.0,
            "num_walls": 0,
            "wall_configs": [],
            "num_random_obstacles": 0,
            "random_obstacle_configs": [],
        }
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
            vision_attributes=vision_attributes,
            output_folder=output_folder,
            ceiling_height=ceiling_height,
            wall_x_offset=wall_x_offset,
        )
        # Keep LiDAR behavior identical to the main simulation except mount pitch.
        self.LIDAR3D_MOUNT_PITCH_DEG = 9.5

    @staticmethod
    def _sample_xy_in_radius(rng: np.random.Generator, radius: float) -> list[float]:
        """Uniform sample in a disk centered at origin."""
        r = float(radius) * float(np.sqrt(rng.uniform(0.0, 1.0)))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        return [r * float(np.cos(theta)), r * float(np.sin(theta))]

    @staticmethod
    def sample_random_scene_configuration(
        rng: np.random.Generator,
        enable_whiteboard: bool | None = None,
        enable_distractor: bool | None = None,
        enable_ceiling: bool | None = None,
        num_walls: int | None = None,
        num_random_obstacles: int | None = None,
    ) -> dict:
        """Sample deterministic scene config from RNG."""
        wb_enabled = bool(rng.random() < 0.75) if enable_whiteboard is None else bool(enable_whiteboard)
        dist_enabled = bool(rng.random() < 0.60) if enable_distractor is None else bool(enable_distractor)
        # Plate-like distractors (ceiling/walls) are disabled by default to avoid
        # over-regular planar clutter in the training distribution.
        ceil_enabled = False if enable_ceiling is None else bool(enable_ceiling)
        n_walls = 0 if num_walls is None else int(num_walls)
        n_rand_obs = int(rng.integers(0, 6)) if num_random_obstacles is None else int(num_random_obstacles)

        wall_cfgs = []
        for _ in range(n_walls):
            wall_cfgs.append({
                "pos_xy": EmptyWallWhiteboardAviary._sample_xy_in_radius(rng, radius=3.0),
                "yaw": float(rng.uniform(0.0, 2.0 * np.pi)),
                "width": float(rng.uniform(1.0, 4.0)),
                "height": float(rng.uniform(1.5, 3.0)),
                "thickness": float(rng.uniform(0.03, 0.10)),
            })

        rand_obs_cfgs = []
        for _ in range(n_rand_obs):
            shape = str(rng.choice(["box", "sphere"]))
            if shape == "sphere":
                radius = float(rng.uniform(0.04, 0.90))
                rand_obs_cfgs.append({
                    "pos_xy": EmptyWallWhiteboardAviary._sample_xy_in_radius(rng, radius=3.0),
                    "z": float(rng.uniform(radius, 2.8)),
                    "shape": "sphere",
                    "radius": radius,
                    # Keep legacy key for compatibility with old readers.
                    "size": float(2.0 * radius),
                })
            else:
                # Anisotropic cuboids: stronger per-axis variation than cubes.
                # Cap any full cuboid dimension at 0.5 m => half-extent <= 0.25 m.
                hx = float(rng.uniform(0.04, 0.25))
                hy = float(rng.uniform(0.04, 0.25))
                hz = float(rng.uniform(0.04, 0.25))
                rand_obs_cfgs.append({
                    "pos_xy": EmptyWallWhiteboardAviary._sample_xy_in_radius(rng, radius=3.0),
                    "z": float(rng.uniform(hz, 2.8)),
                    "shape": "box",
                    "half_extents": [hx, hy, hz],
                    # Keep legacy key for compatibility with old readers.
                    "size": float(2.0 * max(hx, hy, hz)),
                })

        return {
            "enable_whiteboard": wb_enabled,
            "whiteboard_width": float(rng.uniform(1.80, 1.90)),
            "whiteboard_height": float(rng.uniform(1.10, 1.30)),
            "whiteboard_depth": float(rng.uniform(0.02, 0.03)),
            "whiteboard_bottom_z": float(rng.uniform(0.70, 0.80)),
            "pole_side": 0.05,
            "whiteboard_pos_xy": EmptyWallWhiteboardAviary._sample_xy_in_radius(rng, radius=3.0),
            "whiteboard_yaw": float(rng.uniform(0.0, 2.0 * np.pi)),
            "enable_distractor": dist_enabled,
            "distractor_pos_xy": EmptyWallWhiteboardAviary._sample_xy_in_radius(rng, radius=3.0),
            "distractor_size": float(rng.uniform(0.10, 0.40)),
            "distractor_height_z": float(rng.uniform(0.3, 2.0)),
            "distractor_shape": str(rng.choice(["box", "cylinder"])),
            "enable_ceiling": ceil_enabled,
            # Roof plane Z (m); keep at least 4 m for LiDAR / drone headroom in collection.
            "ceiling_height": float(rng.uniform(4.0, 5.0)),
            "num_walls": n_walls,
            "wall_configs": wall_cfgs,
            "num_random_obstacles": n_rand_obs,
            "random_obstacle_configs": rand_obs_cfgs,
        }

    def set_scene_configuration(self, **kwargs):
        """Update scene configuration used by the next reset/build."""
        for key, value in kwargs.items():
            if key in self.SCENE_CONFIG:
                self.SCENE_CONFIG[key] = value
            elif key.startswith(("enable_lightswitch", "lightswitch_")):
                pass  # silently ignore legacy lightswitch keys
            else:
                raise KeyError(f"Unknown scene config key: {key}")

    def _actionSpace(self):
        """Returns the action space of the environment."""
        act_lower_bound = np.array([[0.0, 0.0, 0.0, 0.0] for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array(
            [[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for _ in range(self.NUM_DRONES)]
        )
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """Returns the observation space of the environment."""
        obs_lower_bound = np.array(
            [
                [
                    -np.inf,
                    -np.inf,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -np.pi,
                    -np.pi,
                    -np.pi,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                for _ in range(self.NUM_DRONES)
            ]
        )
        obs_upper_bound = np.array(
            [
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    np.pi,
                    np.pi,
                    np.pi,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    self.MAX_RPM,
                    self.MAX_RPM,
                    self.MAX_RPM,
                    self.MAX_RPM,
                ]
                for _ in range(self.NUM_DRONES)
            ]
        )
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment."""
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs."""
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    def _computeReward(self):
        """Unused for this control-oriented environment."""
        return -1

    def _computeTerminated(self):
        """Unused for this control-oriented environment."""
        return False

    def _computeTruncated(self):
        """Unused for this control-oriented environment."""
        return False

    def _computeInfo(self):
        """Unused for this control-oriented environment."""
        return {"answer": 42}

    def _housekeeping(self):
        """Initialize simulation and create only the requested minimal scene."""
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))

        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array(
            [
                p.loadURDF(
                    pkg_resources.resource_filename("gym_pybullet_drones", "assets/" + self.URDF),
                    self.INIT_XYZS[i, :],
                    p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                    flags=p.URDF_USE_INERTIA_FROM_FILE,
                    physicsClientId=self.CLIENT,
                )
                for i in range(self.NUM_DRONES)
            ]
        )

        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)

        self._buildMinimalScene()

    def _buildMinimalScene(self):
        """Create whiteboard and distractor obstacles."""
        cfg = self.SCENE_CONFIG
        self.MINIMAL_WALL_ID = None
        self.WHITEBOARD_ID = None
        self.WHITEBOARD_POLE_IDS = []
        self.DISTRACTOR_PART_IDS = []
        # Legacy alias so old code referencing LIGHTSWITCH_PART_IDS still works.
        self.LIGHTSWITCH_PART_IDS = self.DISTRACTOR_PART_IDS

        whiteboard_width = float(cfg["whiteboard_width"])
        whiteboard_height = float(cfg["whiteboard_height"])
        whiteboard_depth = float(cfg["whiteboard_depth"])
        whiteboard_bottom_z = float(cfg["whiteboard_bottom_z"])
        pole_side = float(cfg["pole_side"])
        whiteboard_pos_xy = np.asarray(cfg["whiteboard_pos_xy"], dtype=np.float64)
        whiteboard_yaw = float(cfg["whiteboard_yaw"])
        enable_whiteboard = bool(cfg["enable_whiteboard"])

        if enable_whiteboard:
            whiteboard_half_extents = [
                whiteboard_depth / 2.0,
                whiteboard_width / 2.0,
                whiteboard_height / 2.0,
            ]
            whiteboard_collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=whiteboard_half_extents, physicsClientId=self.CLIENT
            )
            whiteboard_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=whiteboard_half_extents,
                rgbaColor=[0.98, 0.98, 0.98, 1.0],
                physicsClientId=self.CLIENT,
            )
            whiteboard_center_z = whiteboard_bottom_z + whiteboard_height / 2.0
            self.WHITEBOARD_ID = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=whiteboard_collision,
                baseVisualShapeIndex=whiteboard_visual,
                basePosition=[float(whiteboard_pos_xy[0]), float(whiteboard_pos_xy[1]), whiteboard_center_z],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, whiteboard_yaw]),
                physicsClientId=self.CLIENT,
            )

            pole_height = whiteboard_bottom_z
            pole_half_extents = [pole_side / 2.0, pole_side / 2.0, pole_height / 2.0]
            pole_collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=pole_half_extents, physicsClientId=self.CLIENT
            )
            pole_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=pole_half_extents,
                rgbaColor=[0.30, 0.30, 0.30, 1.0],
                physicsClientId=self.CLIENT,
            )

            pole_offset_y = (whiteboard_width / 2.0) - (pole_side / 2.0)
            pole_center_z = pole_height / 2.0
            local_offsets = np.array(
                [[0.0, -pole_offset_y, pole_center_z], [0.0, pole_offset_y, pole_center_z]],
                dtype=np.float64,
            )
            rot = np.array(
                [
                    [np.cos(whiteboard_yaw), -np.sin(whiteboard_yaw), 0.0],
                    [np.sin(whiteboard_yaw), np.cos(whiteboard_yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            for offset in local_offsets:
                world = np.array([whiteboard_pos_xy[0], whiteboard_pos_xy[1], 0.0], dtype=np.float64) + rot @ offset
                pole_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=pole_collision,
                    baseVisualShapeIndex=pole_visual,
                    basePosition=world.tolist(),
                    physicsClientId=self.CLIENT,
                )
                self.WHITEBOARD_POLE_IDS.append(pole_id)

        # --- Simple distractor obstacle (replaces old lightswitch) ---
        if bool(cfg.get("enable_distractor", False)):
            d_xy = np.asarray(cfg["distractor_pos_xy"], dtype=np.float64)
            d_size = float(cfg["distractor_size"])
            d_z = float(cfg["distractor_height_z"])
            d_shape = str(cfg.get("distractor_shape", "box"))
            half = d_size / 2.0
            color = [0.55, 0.45, 0.35, 1.0]
            if d_shape == "cylinder":
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=half, height=d_size, physicsClientId=self.CLIENT)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=half, length=d_size, rgbaColor=color, physicsClientId=self.CLIENT)
            else:
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half], physicsClientId=self.CLIENT)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=color, physicsClientId=self.CLIENT)
            bid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                    basePosition=[float(d_xy[0]), float(d_xy[1]), d_z], physicsClientId=self.CLIENT)
            self.DISTRACTOR_PART_IDS.append(int(bid))

        # --- Ceiling ---
        if bool(cfg.get("enable_ceiling", False)):
            ch = max(4.0, float(cfg.get("ceiling_height", 4.0)))
            arena_half = 6.0
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_half, arena_half, 0.02], physicsClientId=self.CLIENT)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_half, arena_half, 0.02],
                                      rgbaColor=[0.85, 0.85, 0.85, 1.0], physicsClientId=self.CLIENT)
            bid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                    basePosition=[0.0, 0.0, ch], physicsClientId=self.CLIENT)
            self.DISTRACTOR_PART_IDS.append(int(bid))

        # --- Wall segments ---
        for wc in cfg.get("wall_configs", []):
            w_xy = np.asarray(wc["pos_xy"], dtype=np.float64)
            w_yaw = float(wc["yaw"])
            w_width = float(wc["width"])
            w_height = float(wc["height"])
            w_thick = float(wc["thickness"])
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w_thick / 2.0, w_width / 2.0, w_height / 2.0],
                                         physicsClientId=self.CLIENT)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w_thick / 2.0, w_width / 2.0, w_height / 2.0],
                                      rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=self.CLIENT)
            bid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                    basePosition=[float(w_xy[0]), float(w_xy[1]), w_height / 2.0],
                                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, w_yaw]),
                                    physicsClientId=self.CLIENT)
            self.DISTRACTOR_PART_IDS.append(int(bid))

        # --- Random small obstacles (cubes / spheres) ---
        for oc in cfg.get("random_obstacle_configs", []):
            o_xy = np.asarray(oc["pos_xy"], dtype=np.float64)
            o_z = float(oc["z"])
            o_size = float(oc.get("size", 0.25))
            o_shape = str(oc.get("shape", "box"))
            r_color = [float(np.clip(0.3 + 0.5 * abs(hash(str(oc)) % 100) / 100.0, 0.2, 0.9))] * 3 + [1.0]
            if o_shape == "sphere":
                radius = float(oc.get("radius", o_size / 2.0))
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.CLIENT)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=r_color, physicsClientId=self.CLIENT)
            else:
                half_ext = oc.get("half_extents", None)
                if half_ext is None:
                    half = o_size / 2.0
                    half_ext = [half, half, half]
                hx, hy, hz = [float(x) for x in half_ext]
                # Safety clamp: ensure any full cuboid dimension never exceeds 0.5 m.
                hx = float(np.clip(hx, 1e-3, 0.25))
                hy = float(np.clip(hy, 1e-3, 0.25))
                hz = float(np.clip(hz, 1e-3, 0.25))
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self.CLIENT)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=r_color, physicsClientId=self.CLIENT)
            bid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                    basePosition=[float(o_xy[0]), float(o_xy[1]), o_z], physicsClientId=self.CLIENT)
            self.DISTRACTOR_PART_IDS.append(int(bid))

    def _getDroneLidarScan3D(self, nth_drone, max_range=None, return_point_cloud=False, return_hit_ids=False):
        """Return 3D LiDAR scan directly at output resolution (no oversample/downsample)."""
        if max_range is None:
            max_range = self.LIDAR3D_MAX_RANGE

        num_beams = self.LIDAR3D_NUM_BEAMS
        num_bins = self.LIDAR3D_NUM_BINS
        elevation_angles = np.linspace(0, np.deg2rad(self.LIDAR3D_VERTICAL_FOV), num_beams, endpoint=True)
        azimuth_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
        az_grid, el_grid = np.meshgrid(azimuth_angles, elevation_angles, indexing="xy")
        az_flat = az_grid.flatten()
        el_flat = el_grid.flatten()

        drone_pos = np.array(self.pos[nth_drone, :])
        drone_quat = self.quat[nth_drone, :]
        rot_mat = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

        lidar_z_offset = self.COLLISION_H / 2 + self.COLLISION_Z_OFFSET + 0.05
        lidar_origin_body = np.array([0, 0, lidar_z_offset])
        lidar_origin_world = drone_pos + rot_mat @ lidar_origin_body

        cos_el = np.cos(el_flat)
        sin_el = np.sin(el_flat)
        cos_az = np.cos(az_flat)
        sin_az = np.sin(az_flat)
        ray_dirs_local = np.column_stack([cos_el * cos_az, cos_el * sin_az, sin_el])

        lidar_pitch_rad = np.deg2rad(float(self.LIDAR3D_MOUNT_PITCH_DEG))
        cos_pitch = np.cos(lidar_pitch_rad)
        sin_pitch = np.sin(lidar_pitch_rad)
        pitch_rotation = np.array(
            [
                [cos_pitch, 0, sin_pitch],
                [0, 1, 0],
                [-sin_pitch, 0, cos_pitch],
            ]
        )
        ray_dirs_local = (pitch_rotation @ ray_dirs_local.T).T

        ray_dirs_world = (rot_mat @ ray_dirs_local.T).T
        total_rays = num_beams * num_bins
        ray_from = np.tile(lidar_origin_world, (total_rays, 1))
        ray_to = ray_from + ray_dirs_world * max_range

        p.performCollisionDetection(physicsClientId=self.CLIENT)

        # PyBullet caps rayTestBatch at 16384 rays; split into chunks.
        MAX_RAYS_PER_BATCH = 16384
        ray_hits = []
        for start in range(0, total_rays, MAX_RAYS_PER_BATCH):
            end = min(start + MAX_RAYS_PER_BATCH, total_rays)
            batch = p.rayTestBatch(
                rayFromPositions=ray_from[start:end].tolist(),
                rayToPositions=ray_to[start:end].tolist(),
                parentObjectUniqueId=-1,
                physicsClientId=self.CLIENT,
            )
            ray_hits.extend(batch)

        range_out = np.ones((num_beams, num_bins), dtype=np.float32)
        mask_out = np.zeros((num_beams, num_bins), dtype=np.float32)
        ids_out = -1 * np.ones((num_beams, num_bins), dtype=np.int32)
        drone_id = int(self.DRONE_IDS[nth_drone])

        for i, hit in enumerate(ray_hits):
            e = i // num_bins
            a = i % num_bins
            if hit[0] != -1 and hit[0] != drone_id:
                hit_distance = hit[2] * max_range
                range_out[e, a] = min(hit_distance / max_range, 1.0)
                mask_out[e, a] = 1.0
                ids_out[e, a] = int(hit[0])

        range_image = np.stack([range_out, mask_out], axis=-1).astype(np.float32)

        if return_point_cloud:
            hit_points, ranges, ray_angles = self._range_image_to_point_cloud(
                range_image, elevation_angles, azimuth_angles, max_range, pitch_rotation
            )
            if return_hit_ids:
                return range_image, hit_points, ranges, ray_angles, ids_out
            return range_image, hit_points, ranges, ray_angles
        if return_hit_ids:
            return range_image, ids_out
        return range_image
