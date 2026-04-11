import pkg_resources
import time
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Whiteboard dimension ranges (kept identical to collect_realistic_latent.py RANGES).
_WB_WIDTH_RANGE = (1.70, 2.00)
_WB_HEIGHT_RANGE = (1.10, 1.30)
_WB_BOTTOM_Z_RANGE = (0.70, 0.90)
_WB_DEPTH_RANGE = (0.02, 0.04)
_WB_POLE_SIDE = 0.05

# Minimum centre-to-centre distance between whiteboards (m).
_WB_MIN_SEPARATION_M = 2.0

# Arena half-side for the hw-demo 8 × 8 square room.
HW_DEMO_HALF_SIDE_M = 4.0


class HwDemoAviary(BaseAviary):
    """
    Control-oriented environment for the hardware demo.

    Key differences from EmptyWallWhiteboardAviary:
    - Supports 0-3 whiteboards (previously single board).
    - Drone and objects spawn inside an 8×8 m square instead of a radius-3 disk.
    - No ceiling and no wall segments by default (open outdoor-style arena).
    - SCENE_CONFIG uses a ``whiteboards`` list; legacy single-board keys are gone.
    """

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
            # List of per-board dicts; each dict has the keys listed below.
            # Board dict keys: pos_xy, yaw, width, height, depth, bottom_z, pole_side.
            "whiteboards": [],
            "enable_distractor": False,
            "distractor_pos_xy": [1.0, 1.0],
            "distractor_size": 0.25,
            "distractor_height_z": 1.0,
            "distractor_shape": "box",
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
        # Keep LiDAR behaviour identical to the main simulation except mount pitch.
        self.LIDAR3D_MOUNT_PITCH_DEG = 9.5

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_xy_in_square(
        rng: np.random.Generator,
        half_side: float = HW_DEMO_HALF_SIDE_M,
    ) -> list[float]:
        """Uniform sample inside an axis-aligned square centred at origin."""
        return [
            float(rng.uniform(-half_side, half_side)),
            float(rng.uniform(-half_side, half_side)),
        ]

    @staticmethod
    def _sample_xy_in_radius(rng: np.random.Generator, radius: float) -> list[float]:
        """Uniform sample in a disk centred at origin (kept for backward compat)."""
        r = float(radius) * float(np.sqrt(rng.uniform(0.0, 1.0)))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        return [r * float(np.cos(theta)), r * float(np.sin(theta))]

    # ------------------------------------------------------------------
    # Scene configuration
    # ------------------------------------------------------------------

    @staticmethod
    def sample_random_scene_configuration(
        rng: np.random.Generator,
        num_whiteboards: int | None = None,
        num_random_obstacles: int | None = None,
    ) -> dict:
        """
        Sample a deterministic scene configuration from *rng*.

        Parameters
        ----------
        rng:
            NumPy random generator.
        num_whiteboards:
            Override number of whiteboards (0-3). If None, sampled uniformly in {0,1,2,3}.
        num_random_obstacles:
            Override number of small random obstacles. If None, sampled in {0,...,5}.
        """
        n_boards = (
            int(rng.integers(0, 4))
            if num_whiteboards is None
            else int(np.clip(num_whiteboards, 0, 3))
        )
        n_rand_obs = (
            int(rng.integers(0, 6))
            if num_random_obstacles is None
            else int(num_random_obstacles)
        )

        # --- Whiteboards ---
        boards: list[dict] = []
        for _ in range(n_boards):
            for _attempt in range(100):
                xy = HwDemoAviary._sample_xy_in_square(rng, HW_DEMO_HALF_SIDE_M)
                # Enforce minimum separation from existing boards.
                too_close = any(
                    np.hypot(xy[0] - b["pos_xy"][0], xy[1] - b["pos_xy"][1])
                    < _WB_MIN_SEPARATION_M
                    for b in boards
                )
                if not too_close:
                    break
            boards.append({
                "pos_xy": xy,
                "yaw": float(rng.uniform(0.0, 2.0 * np.pi)),
                "width": float(rng.uniform(*_WB_WIDTH_RANGE)),
                "height": float(rng.uniform(*_WB_HEIGHT_RANGE)),
                "depth": float(rng.uniform(*_WB_DEPTH_RANGE)),
                "bottom_z": float(rng.uniform(*_WB_BOTTOM_Z_RANGE)),
                "pole_side": _WB_POLE_SIDE,
            })

        # --- Random small obstacles ---
        rand_obs_cfgs: list[dict] = []
        for _ in range(n_rand_obs):
            shape = str(rng.choice(["box", "sphere"]))
            if shape == "sphere":
                radius = float(rng.uniform(0.04, 0.90))
                rand_obs_cfgs.append({
                    "pos_xy": HwDemoAviary._sample_xy_in_square(rng, HW_DEMO_HALF_SIDE_M),
                    "z": float(rng.uniform(radius, 2.8)),
                    "shape": "sphere",
                    "radius": radius,
                    "size": float(2.0 * radius),
                })
            else:
                hx = float(rng.uniform(0.04, 0.25))
                hy = float(rng.uniform(0.04, 0.25))
                hz = float(rng.uniform(0.04, 0.25))
                rand_obs_cfgs.append({
                    "pos_xy": HwDemoAviary._sample_xy_in_square(rng, HW_DEMO_HALF_SIDE_M),
                    "z": float(rng.uniform(hz, 2.8)),
                    "shape": "box",
                    "half_extents": [hx, hy, hz],
                    "size": float(2.0 * max(hx, hy, hz)),
                })

        return {
            "whiteboards": boards,
            "enable_distractor": False,
            "num_random_obstacles": n_rand_obs,
            "random_obstacle_configs": rand_obs_cfgs,
        }

    def set_scene_configuration(self, **kwargs):
        """Update scene configuration used by the next reset / build."""
        for key, value in kwargs.items():
            if key in self.SCENE_CONFIG:
                self.SCENE_CONFIG[key] = value
            elif key.startswith(("enable_lightswitch", "lightswitch_")):
                pass  # silently ignore legacy keys
            else:
                raise KeyError(f"Unknown scene config key: {key!r}")

    # ------------------------------------------------------------------
    # Gym / BaseAviary boilerplate
    # ------------------------------------------------------------------

    def _actionSpace(self):
        act_lower = np.array([[0.0, 0.0, 0.0, 0.0] for _ in range(self.NUM_DRONES)])
        act_upper = np.array(
            [[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM]
             for _ in range(self.NUM_DRONES)]
        )
        return spaces.Box(low=act_lower, high=act_upper, dtype=np.float32)

    def _observationSpace(self):
        low = np.array([
            [-np.inf, -np.inf, 0.0,
             -1.0, -1.0, -1.0, -1.0,
             -np.pi, -np.pi, -np.pi,
             -np.inf, -np.inf, -np.inf,
             -np.inf, -np.inf, -np.inf,
             0.0, 0.0, 0.0, 0.0]
            for _ in range(self.NUM_DRONES)
        ])
        high = np.array([
            [np.inf, np.inf, np.inf,
             1.0, 1.0, 1.0, 1.0,
             np.pi, np.pi, np.pi,
             np.inf, np.inf, np.inf,
             np.inf, np.inf, np.inf,
             self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM]
            for _ in range(self.NUM_DRONES)
        ])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    def _preprocessAction(self, action):
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    def _computeReward(self):
        return -1

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {"answer": 42}

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _housekeeping(self):
        """Initialise simulation and build the scene."""
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
        self.DRONE_IDS = np.array([
            p.loadURDF(
                pkg_resources.resource_filename("gym_pybullet_drones", "assets/" + self.URDF),
                self.INIT_XYZS[i, :],
                p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.CLIENT,
            )
            for i in range(self.NUM_DRONES)
        ])

        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)

        self._buildMinimalScene()

    def _buildMinimalScene(self):
        """Spawn all whiteboards and random obstacles defined in SCENE_CONFIG."""
        cfg = self.SCENE_CONFIG

        # --- Whiteboard tracking ---
        self.WHITEBOARD_IDS: list[int] = []
        self.WHITEBOARD_POLE_IDS: list[int] = []
        # Expose the live whiteboard config list so tasks can query geometry.
        self.whiteboard_configs: list[dict] = cfg.get("whiteboards", [])

        for wb in self.whiteboard_configs:
            self._spawn_whiteboard(wb)

        # Backward-compat single-board alias.
        self.WHITEBOARD_ID: int | None = self.WHITEBOARD_IDS[0] if self.WHITEBOARD_IDS else None

        # --- Distractor obstacle (optional) ---
        self.DISTRACTOR_PART_IDS: list[int] = []
        self.LIGHTSWITCH_PART_IDS = self.DISTRACTOR_PART_IDS  # legacy alias

        if bool(cfg.get("enable_distractor", False)):
            d_xy = np.asarray(cfg["distractor_pos_xy"], dtype=np.float64)
            d_size = float(cfg["distractor_size"])
            d_z = float(cfg["distractor_height_z"])
            d_shape = str(cfg.get("distractor_shape", "box"))
            half = d_size / 2.0
            color = [0.55, 0.45, 0.35, 1.0]
            if d_shape == "cylinder":
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=half, height=d_size, physicsClientId=self.CLIENT
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=half, length=d_size, rgbaColor=color,
                    physicsClientId=self.CLIENT,
                )
            else:
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[half, half, half], physicsClientId=self.CLIENT
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=color,
                    physicsClientId=self.CLIENT,
                )
            bid = p.createMultiBody(
                baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                basePosition=[float(d_xy[0]), float(d_xy[1]), d_z],
                physicsClientId=self.CLIENT,
            )
            self.DISTRACTOR_PART_IDS.append(int(bid))

        # --- Random small obstacles ---
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
                    half_scalar = o_size / 2.0
                    half_ext = [half_scalar, half_scalar, half_scalar]
                hx, hy, hz = [float(x) for x in half_ext]
                hx = float(np.clip(hx, 1e-3, 0.25))
                hy = float(np.clip(hy, 1e-3, 0.25))
                hz = float(np.clip(hz, 1e-3, 0.25))
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self.CLIENT)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=r_color, physicsClientId=self.CLIENT)
            bid = p.createMultiBody(
                baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                basePosition=[float(o_xy[0]), float(o_xy[1]), o_z],
                physicsClientId=self.CLIENT,
            )
            self.DISTRACTOR_PART_IDS.append(int(bid))

    def _spawn_whiteboard(self, wb: dict) -> None:
        """Build one whiteboard (board body + two support poles) from a board dict."""
        width = float(wb["width"])
        height = float(wb["height"])
        depth = float(wb.get("depth", 0.025))
        bottom_z = float(wb["bottom_z"])
        pole_side = float(wb.get("pole_side", _WB_POLE_SIDE))
        pos_xy = np.asarray(wb["pos_xy"], dtype=np.float64)
        yaw = float(wb["yaw"])

        # Board body — depth along local X, width along local Y, height along Z.
        half_extents = [depth / 2.0, width / 2.0, height / 2.0]
        wb_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.CLIENT
        )
        wb_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[0.98, 0.98, 0.98, 1.0], physicsClientId=self.CLIENT,
        )
        center_z = bottom_z + height / 2.0
        wb_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=wb_col,
            baseVisualShapeIndex=wb_vis,
            basePosition=[float(pos_xy[0]), float(pos_xy[1]), center_z],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            physicsClientId=self.CLIENT,
        )
        self.WHITEBOARD_IDS.append(int(wb_id))

        # Support poles (two vertical pillars flanking the board).
        pole_height = bottom_z
        pole_half = [pole_side / 2.0, pole_side / 2.0, pole_height / 2.0]
        pole_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=pole_half, physicsClientId=self.CLIENT
        )
        pole_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=pole_half,
            rgbaColor=[0.30, 0.30, 0.30, 1.0], physicsClientId=self.CLIENT,
        )
        pole_offset_y = (width / 2.0) - (pole_side / 2.0)
        pole_center_z = pole_height / 2.0
        rot = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0,          0.0,         1.0],
        ], dtype=np.float64)
        for s in (-1.0, 1.0):
            local_offset = np.array([0.0, s * pole_offset_y, pole_center_z], dtype=np.float64)
            world = np.array([pos_xy[0], pos_xy[1], 0.0]) + rot @ local_offset
            pole_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=pole_col,
                baseVisualShapeIndex=pole_vis,
                basePosition=world.tolist(),
                physicsClientId=self.CLIENT,
            )
            self.WHITEBOARD_POLE_IDS.append(int(pole_id))

    # ------------------------------------------------------------------
    # LiDAR sensor
    # ------------------------------------------------------------------

    def _getDroneLidarScan3D(
        self,
        nth_drone,
        max_range=None,
        return_point_cloud=False,
        return_hit_ids=False,
    ):
        """Return a 3D LiDAR scan at output resolution (no oversample / downsample)."""
        if max_range is None:
            max_range = self.LIDAR3D_MAX_RANGE

        num_beams = self.LIDAR3D_NUM_BEAMS
        num_bins = self.LIDAR3D_NUM_BINS
        elevation_angles = np.linspace(
            0, np.deg2rad(self.LIDAR3D_VERTICAL_FOV), num_beams, endpoint=True
        )
        azimuth_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
        az_grid, el_grid = np.meshgrid(azimuth_angles, elevation_angles, indexing="xy")
        az_flat = az_grid.flatten()
        el_flat = el_grid.flatten()

        drone_pos = np.array(self.pos[nth_drone, :])
        drone_quat = self.quat[nth_drone, :]
        rot_mat = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

        lidar_z_offset = self.COLLISION_H / 2 + self.COLLISION_Z_OFFSET + 0.05
        lidar_origin_world = drone_pos + rot_mat @ np.array([0, 0, lidar_z_offset])

        cos_el = np.cos(el_flat)
        sin_el = np.sin(el_flat)
        cos_az = np.cos(az_flat)
        sin_az = np.sin(az_flat)
        ray_dirs_local = np.column_stack([cos_el * cos_az, cos_el * sin_az, sin_el])

        lidar_pitch_rad = np.deg2rad(float(self.LIDAR3D_MOUNT_PITCH_DEG))
        cos_pitch = np.cos(lidar_pitch_rad)
        sin_pitch = np.sin(lidar_pitch_rad)
        pitch_rotation = np.array([
            [cos_pitch, 0, sin_pitch],
            [0,         1, 0        ],
            [-sin_pitch,0, cos_pitch],
        ])
        ray_dirs_local = (pitch_rotation @ ray_dirs_local.T).T
        ray_dirs_world = (rot_mat @ ray_dirs_local.T).T

        total_rays = num_beams * num_bins
        ray_from = np.tile(lidar_origin_world, (total_rays, 1))
        ray_to = ray_from + ray_dirs_world * max_range

        p.performCollisionDetection(physicsClientId=self.CLIENT)

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


# Backward-compatibility alias so existing scripts that import
# EmptyWallWhiteboardAviary continue to work unchanged.
EmptyWallWhiteboardAviary = HwDemoAviary
