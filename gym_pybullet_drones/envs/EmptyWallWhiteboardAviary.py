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
        ceiling_height: float = 2.5,
        wall_x_offset: float = 3.0,
    ):
        self.SCENE_CONFIG = {
            "seed": 0,
            "enable_whiteboard": True,
            "enable_lightswitch": True,
            "whiteboard_width": 1.84,
            "whiteboard_height": 1.20,
            "whiteboard_depth": 0.025,
            "whiteboard_bottom_z": 0.75,
            "pole_side": 0.05,
            "whiteboard_pos_xy": [0.0, 0.0],
            "whiteboard_yaw": 0.0,
            "lightswitch_pos_xy": [1.0, 1.0],
            "lightswitch_yaw": 0.0,
            "lightswitch_panel_size": 0.25,
            "lightswitch_panel_thickness": 0.05,
            "lightswitch_panel_tilt_deg": 45.0,
            "lightswitch_panel_forward_offset": 0.088,
            "lightswitch_tripod_height": 1.30,
            "lightswitch_leg_diameter": 0.05,
            "lightswitch_leg_tilt_deg": 25.0,
            "lightswitch_back_strut_diameter": 0.05,
            "lightswitch_back_strut_length": 0.50,
            "lightswitch_connection_offset_y": 0.0,
            "lightswitch_connection_offset_z": 0.0,
            "lightswitch_cube_size": 0.07,
            "lightswitch_cube_depth": 0.05,
            "lightswitch_cube_bottom_offset": 0.03,
            "lightswitch_cone_diameter": 0.05,
            "lightswitch_cone_depth": 0.10,
            "lightswitch_cone_top_offset": 0.05,
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
    def _quat_from_z_axis(direction: np.ndarray):
        """Quaternion rotating +Z to `direction`."""
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        if dot > 1.0 - 1e-8:
            return [0.0, 0.0, 0.0, 1.0]
        if dot < -1.0 + 1e-8:
            return p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
        axis = np.cross(z_axis, direction)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        angle = float(np.arccos(dot))
        return p.getQuaternionFromAxisAngle(axis.tolist(), angle)

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
        enable_lightswitch: bool | None = None,
    ) -> dict:
        """Sample deterministic scene config from RNG."""
        # Independent toggles: each object appears with 75% probability by default.
        wb_enabled = bool(rng.random() < 0.75) if enable_whiteboard is None else bool(enable_whiteboard)
        ls_enabled = bool(rng.random() < 0.75) if enable_lightswitch is None else bool(enable_lightswitch)
        return {
            "enable_whiteboard": wb_enabled,
            "enable_lightswitch": ls_enabled,
            "whiteboard_width": float(rng.uniform(1.80, 1.90)),
            "whiteboard_height": float(rng.uniform(1.10, 1.30)),
            "whiteboard_depth": float(rng.uniform(0.02, 0.03)),
            "whiteboard_bottom_z": float(rng.uniform(0.70, 0.80)),
            "pole_side": 0.05,
            "whiteboard_pos_xy": EmptyWallWhiteboardAviary._sample_xy_in_radius(rng, radius=5.0),
            "whiteboard_yaw": float(rng.uniform(0.0, 2.0 * np.pi)),
            "lightswitch_pos_xy": EmptyWallWhiteboardAviary._sample_xy_in_radius(rng, radius=5.0),
            "lightswitch_yaw": float(rng.uniform(0.0, 2.0 * np.pi)),
            "lightswitch_panel_size": float(rng.uniform(0.20, 0.35)),
            "lightswitch_panel_thickness": float(rng.uniform(0.035, 0.070)),
            "lightswitch_panel_tilt_deg": float(rng.uniform(30.0, 60.0)),
            "lightswitch_panel_forward_offset": float(rng.uniform(0.04, 0.16)),
            "lightswitch_tripod_height": float(rng.uniform(1.00, 1.60)),
            "lightswitch_leg_diameter": float(rng.uniform(0.03, 0.07)),
            # Same tilt for all three legs; wider range for domain randomization.
            "lightswitch_leg_tilt_deg": float(rng.uniform(7.0, 52.0)),
            "lightswitch_back_strut_diameter": float(rng.uniform(0.035, 0.08)),
            "lightswitch_back_strut_length": float(rng.uniform(0.30, 0.75)),
            "lightswitch_connection_offset_y": float(rng.uniform(-0.04, 0.04)),
            "lightswitch_connection_offset_z": float(rng.uniform(-0.04, 0.04)),
            "lightswitch_cube_size": float(rng.uniform(0.05, 0.10)),
            "lightswitch_cube_depth": float(rng.uniform(0.03, 0.08)),
            "lightswitch_cube_bottom_offset": float(rng.uniform(0.01, 0.09)),
            "lightswitch_cone_diameter": float(rng.uniform(0.035, 0.08)),
            "lightswitch_cone_depth": float(rng.uniform(0.06, 0.16)),
            "lightswitch_cone_top_offset": float(rng.uniform(0.01, 0.09)),
        }

    def set_scene_configuration(self, **kwargs):
        """Update scene configuration used by the next reset/build."""
        for key, value in kwargs.items():
            if key not in self.SCENE_CONFIG:
                raise KeyError(f"Unknown scene config key: {key}")
            self.SCENE_CONFIG[key] = value

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
        """Create randomized whiteboard/lightswitch hardware scene."""
        cfg = self.SCENE_CONFIG
        self.MINIMAL_WALL_ID = None
        self.WHITEBOARD_ID = None
        self.WHITEBOARD_POLE_IDS = []
        self.LIGHTSWITCH_ID = None
        self.LIGHTSWITCH_PART_IDS = []

        whiteboard_width = float(cfg["whiteboard_width"])
        whiteboard_height = float(cfg["whiteboard_height"])
        whiteboard_depth = float(cfg["whiteboard_depth"])
        whiteboard_bottom_z = float(cfg["whiteboard_bottom_z"])
        pole_side = float(cfg["pole_side"])
        whiteboard_pos_xy = np.asarray(cfg["whiteboard_pos_xy"], dtype=np.float64)
        whiteboard_yaw = float(cfg["whiteboard_yaw"])
        enable_whiteboard = bool(cfg["enable_whiteboard"])
        enable_lightswitch = bool(cfg["enable_lightswitch"])

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

        if enable_lightswitch:
            pos_xy = np.asarray(cfg["lightswitch_pos_xy"], dtype=np.float64)
            yaw = float(cfg["lightswitch_yaw"])
            panel_size = float(cfg["lightswitch_panel_size"])
            panel_thickness = float(cfg["lightswitch_panel_thickness"])
            panel_tilt = np.deg2rad(float(cfg["lightswitch_panel_tilt_deg"]))
            panel_forward_offset = float(cfg["lightswitch_panel_forward_offset"])
            tripod_height = float(cfg["lightswitch_tripod_height"])
            leg_radius = float(cfg["lightswitch_leg_diameter"]) / 2.0
            leg_tilt = np.deg2rad(float(cfg["lightswitch_leg_tilt_deg"]))
            back_strut_radius = float(cfg["lightswitch_back_strut_diameter"]) / 2.0
            back_strut_length = float(cfg["lightswitch_back_strut_length"])
            connection_offset_y = float(cfg["lightswitch_connection_offset_y"])
            connection_offset_z = float(cfg["lightswitch_connection_offset_z"])

            cube_size = float(cfg["lightswitch_cube_size"])
            cube_depth = float(cfg["lightswitch_cube_depth"])
            cube_bottom_offset = float(cfg["lightswitch_cube_bottom_offset"])
            cone_radius = float(cfg["lightswitch_cone_diameter"]) / 2.0
            cone_depth = float(cfg["lightswitch_cone_depth"])
            cone_top_offset = float(cfg["lightswitch_cone_top_offset"])

            # Strong randomization with coherence guards.
            panel_size = max(panel_size, 0.16)
            panel_thickness = float(np.clip(panel_thickness, 0.015, panel_size * 0.40))
            tripod_height = max(tripod_height, 0.35)
            leg_radius = max(leg_radius, 0.005)
            leg_tilt = float(np.clip(leg_tilt, np.deg2rad(5.0), np.deg2rad(55.0)))
            back_strut_radius = max(back_strut_radius, 0.005)
            back_strut_length = max(back_strut_length, 0.05)
            cube_size = float(np.clip(cube_size, 0.03, panel_size * 0.55))
            cube_depth = float(np.clip(cube_depth, 0.015, 0.20))
            cone_radius = float(np.clip(cone_radius, 0.01, panel_size * 0.25))
            cone_depth = float(np.clip(cone_depth, 0.03, 0.25))
            max_conn_y = 0.30 * panel_size
            max_conn_z = 0.30 * panel_size
            connection_offset_y = float(np.clip(connection_offset_y, -max_conn_y, max_conn_y))
            connection_offset_z = float(np.clip(connection_offset_z, -max_conn_z, max_conn_z))
            cube_bottom_offset = float(
                np.clip(cube_bottom_offset, 0.0, max(0.0, panel_size - cube_size))
            )
            cone_top_offset = float(
                np.clip(cone_top_offset, 0.0, max(0.0, panel_size - 2.0 * cone_radius))
            )

            panel_center_z = tripod_height + (panel_size / 2.0) * np.cos(panel_tilt)
            panel_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[panel_thickness / 2.0, panel_size / 2.0, panel_size / 2.0],
                physicsClientId=self.CLIENT,
            )
            panel_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[panel_thickness / 2.0, panel_size / 2.0, panel_size / 2.0],
                rgbaColor=[0.55, 0.40, 0.25, 1.0],
                physicsClientId=self.CLIENT,
            )
            panel_quat = p.getQuaternionFromEuler([0.0, panel_tilt, yaw])
            rot_panel = np.array(p.getMatrixFromQuaternion(panel_quat), dtype=np.float64).reshape(3, 3)
            # "Forward" is the plate normal projected on the floor plane (XY only).
            panel_forward_xy = rot_panel @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            panel_forward_xy[2] = 0.0
            norm_xy = max(np.linalg.norm(panel_forward_xy), 1e-9)
            panel_forward_xy = panel_forward_xy / norm_xy
            # Tripod connection reference (unshifted); plate is shifted forward relative to this.
            tripod_panel_origin = np.array(
                [float(pos_xy[0]), float(pos_xy[1]), float(panel_center_z)],
                dtype=np.float64,
            )
            panel_origin = tripod_panel_origin - panel_forward_offset * panel_forward_xy
            self.LIGHTSWITCH_ID = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=panel_collision,
                baseVisualShapeIndex=panel_visual,
                basePosition=panel_origin.tolist(),
                baseOrientation=panel_quat,
                physicsClientId=self.CLIENT,
            )
            self.LIGHTSWITCH_PART_IDS.append(int(self.LIGHTSWITCH_ID))

            cube_center_local = np.array(
                [
                    -panel_thickness / 2.0 - cube_depth / 2.0,
                    connection_offset_y,
                    -panel_size / 2.0 + cube_bottom_offset + cube_size / 2.0,
                ],
                dtype=np.float64,
            )
            cube_center = panel_origin + rot_panel @ cube_center_local
            cube_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[cube_depth / 2.0, cube_size / 2.0, cube_size / 2.0],
                physicsClientId=self.CLIENT,
            )
            cube_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[cube_depth / 2.0, cube_size / 2.0, cube_size / 2.0],
                rgbaColor=[0.75, 0.75, 0.75, 1.0],
                physicsClientId=self.CLIENT,
            )
            cube_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=cube_collision,
                baseVisualShapeIndex=cube_visual,
                basePosition=cube_center.tolist(),
                baseOrientation=panel_quat,
                physicsClientId=self.CLIENT,
            )
            self.LIGHTSWITCH_PART_IDS.append(int(cube_id))

            cone_center_local = np.array(
                [
                    -panel_thickness / 2.0 - cone_depth / 2.0,
                    connection_offset_y,
                    panel_size / 2.0 - cone_top_offset - cone_radius,
                ],
                dtype=np.float64,
            )
            cone_center = panel_origin + rot_panel @ cone_center_local
            cone_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=cone_radius,
                height=cone_depth,
                physicsClientId=self.CLIENT,
            )
            cone_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=cone_radius,
                length=cone_depth,
                rgbaColor=[0.98, 0.98, 0.98, 1.0],
                physicsClientId=self.CLIENT,
            )
            # PyBullet cylinders are aligned with local +Z; rotate so cylinder axis
            # aligns with panel local +X (panel normal), i.e. flat face sits on plank.
            cone_quat = p.multiplyTransforms(
                [0.0, 0.0, 0.0],
                panel_quat,
                [0.0, 0.0, 0.0],
                p.getQuaternionFromEuler([0.0, np.pi / 2.0, 0.0]),
            )[1]
            cone_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=cone_collision,
                baseVisualShapeIndex=cone_visual,
                basePosition=cone_center.tolist(),
                baseOrientation=cone_quat,
                physicsClientId=self.CLIENT,
            )
            self.LIGHTSWITCH_PART_IDS.append(int(cone_id))

            # Keep rear support aligned with the tripod connection center.
            top_anchor = tripod_panel_origin + rot_panel @ np.array(
                [0.0, connection_offset_y, -panel_size / 2.0 + connection_offset_z],
                dtype=np.float64,
            )
            back_anchor = tripod_panel_origin + rot_panel @ np.array(
                [-panel_thickness / 2.0, connection_offset_y, connection_offset_z],
                dtype=np.float64,
            )
            # Vertical strut with top touching the plate underside and extending
            # an additional configured length downward.
            underside_z = float(back_anchor[2])
            strut_top = np.array(
                [float(top_anchor[0]), float(top_anchor[1]), underside_z],
                dtype=np.float64,
            )
            strut_bottom = np.array(
                [float(top_anchor[0]), float(top_anchor[1]), underside_z - back_strut_length],
                dtype=np.float64,
            )
            back_strut_center = 0.5 * (strut_bottom + strut_top)
            back_strut_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=back_strut_radius,
                height=back_strut_length,
                physicsClientId=self.CLIENT,
            )
            back_strut_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=back_strut_radius,
                length=back_strut_length,
                rgbaColor=[0.05, 0.05, 0.05, 1.0],
                physicsClientId=self.CLIENT,
            )
            back_strut_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=back_strut_collision,
                baseVisualShapeIndex=back_strut_visual,
                basePosition=back_strut_center.tolist(),
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.CLIENT,
            )
            self.LIGHTSWITCH_PART_IDS.append(int(back_strut_id))

            # Leg axis has vertical component cos(leg_tilt) (down); feet at z=0 => |leg| = top_z / cos(leg_tilt).
            leg_length = float(top_anchor[2]) / max(np.cos(leg_tilt), 1e-6)
            leg_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=leg_radius,
                height=leg_length,
                physicsClientId=self.CLIENT,
            )
            leg_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=leg_radius,
                length=leg_length,
                rgbaColor=[0.05, 0.05, 0.05, 1.0],
                physicsClientId=self.CLIENT,
            )
            for yaw_off in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
                horiz_dir = np.array([np.cos(yaw + yaw_off), np.sin(yaw + yaw_off), 0.0], dtype=np.float64)
                leg_dir = np.cos(leg_tilt) * np.array([0.0, 0.0, -1.0], dtype=np.float64) + np.sin(leg_tilt) * horiz_dir
                leg_dir = leg_dir / max(np.linalg.norm(leg_dir), 1e-9)
                leg_center = top_anchor + 0.5 * leg_length * leg_dir
                leg_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=leg_collision,
                    baseVisualShapeIndex=leg_visual,
                    basePosition=leg_center.tolist(),
                    baseOrientation=self._quat_from_z_axis(leg_dir),
                    physicsClientId=self.CLIENT,
                )
                self.LIGHTSWITCH_PART_IDS.append(int(leg_id))

    def _getDroneLidarScan3D(self, nth_drone, max_range=None, return_point_cloud=False, return_hit_ids=False):
        """Return 3D LiDAR scan with oversampled ray casting and min-range/max-mask downsampling.

        Casts rays at (num_beams * OS_V, num_bins * OS_H) resolution to match
        the real RoboSense Airy 96-beam sensor pipeline, then pools each
        (OS_V x OS_H) block down to one output pixel using:
          - range: minimum across block (closest surface wins)
          - hit mask: maximum across block (any hit → output is hit)
          - body id: id of the closest-range hit in the block

        With LIDAR3D_OVERSAMPLE_V=1 and LIDAR3D_OVERSAMPLE_H=1 this is
        identical to the original single-resolution implementation.
        """
        if max_range is None:
            max_range = self.LIDAR3D_MAX_RANGE

        num_beams = self.LIDAR3D_NUM_BEAMS
        num_bins = self.LIDAR3D_NUM_BINS
        os_v = self.LIDAR3D_OVERSAMPLE_V
        os_h = self.LIDAR3D_OVERSAMPLE_H
        V_full = num_beams * os_v
        H_full = num_bins * os_h

        elevation_angles_full = np.linspace(0, np.deg2rad(self.LIDAR3D_VERTICAL_FOV), V_full, endpoint=True)
        azimuth_angles_full = np.linspace(0, 2 * np.pi, H_full, endpoint=False)
        az_grid, el_grid = np.meshgrid(azimuth_angles_full, elevation_angles_full, indexing="xy")
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
        total_rays = V_full * H_full
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

        # Build full-resolution arrays
        range_full = np.ones((V_full, H_full), dtype=np.float32)
        mask_full = np.zeros((V_full, H_full), dtype=np.float32)
        ids_full = -1 * np.ones((V_full, H_full), dtype=np.int32)
        drone_id = int(self.DRONE_IDS[nth_drone])

        for i, hit in enumerate(ray_hits):
            e = i // H_full
            a = i % H_full
            if hit[0] != -1 and hit[0] != drone_id:
                hit_distance = hit[2] * max_range
                range_full[e, a] = min(hit_distance / max_range, 1.0)
                mask_full[e, a] = 1.0
                ids_full[e, a] = int(hit[0])

        # ---- Downsample (V_full, H_full) → (num_beams, num_bins) ----
        if os_v == 1 and os_h == 1:
            range_out = range_full
            mask_out = mask_full
            ids_out = ids_full
        else:
            # Range: min per block (closest surface wins)
            range_4d = range_full.reshape(num_beams, os_v, num_bins, os_h)
            range_out = range_4d.min(axis=(1, 3))

            # Hit mask: max per block (any hit → 1)
            mask_4d = mask_full.reshape(num_beams, os_v, num_bins, os_h)
            mask_out = mask_4d.max(axis=(1, 3))

            # Body IDs: pick the ID of the closest hit in each block.
            # Set non-hit ranges to inf so argmin ignores them, then index.
            ids_4d = ids_full.reshape(num_beams, os_v, num_bins, os_h)
            range_for_argmin = np.where(mask_4d > 0.5, range_4d, np.float32(2.0))
            # Merge the two intra-block axes → (num_beams, os_v*os_h, num_bins)
            rfam = range_for_argmin.transpose(0, 2, 1, 3).reshape(num_beams, num_bins, os_v * os_h)
            best = rfam.argmin(axis=2)  # (num_beams, num_bins)
            ids_flat = ids_4d.transpose(0, 2, 1, 3).reshape(num_beams, num_bins, os_v * os_h)
            ids_out = np.take_along_axis(ids_flat, best[..., np.newaxis], axis=2).squeeze(axis=2)
            # Where the whole block had no hits, reset to -1
            ids_out[mask_out < 0.5] = -1

        range_image = np.stack([range_out, mask_out], axis=-1).astype(np.float32)

        if return_point_cloud:
            # For visualization, use output-resolution angles
            elevation_angles_out = np.linspace(0, np.deg2rad(self.LIDAR3D_VERTICAL_FOV), num_beams, endpoint=True)
            azimuth_angles_out = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
            hit_points, ranges, ray_angles = self._range_image_to_point_cloud(
                range_image, elevation_angles_out, azimuth_angles_out, max_range, pitch_rotation
            )
            if return_hit_ids:
                return range_image, hit_points, ranges, ray_angles, ids_out
            return range_image, hit_points, ranges, ray_angles
        if return_hit_ids:
            return range_image, ids_out
        return range_image
