import os
from sys import platform
import time
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType


class BaseAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    # metadata = {'render.modes': ['human']}
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results',
                 ceiling_height: float=3.0,
                 wall_x_offset: float=3
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.
        ceiling_height : float, optional
            Height of the ceiling in meters. If None or <= 0, no ceiling is added. Default is 2.0 meters.
        wall_x_offset : float, optional
            X position in world coordinates where the wall is placed. If None or <= 0, no wall is added. Default is 3.0 meters.

        """
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        self.CEILING_HEIGHT = ceiling_height if ceiling_height and ceiling_height > 0 else None
        self.CEILING_ID = None  # Will be set if ceiling is added
        # For 5-wall setup (4 outer walls + 1 center wall), wall_x_offset is ignored - walls are positioned at ±room_size/2 and x=0
        self.ROOM_SIZE = 15.0  # 15m × 15m room (each wall is 15m long)
        self.WALL_CUBE_IDS = []  # Store outer wall cube IDs (4 walls: North, South, East, West)
        self.CENTER_WALL_CUBE_IDS = []  # Store center wall cube IDs (1 wall at x=0)
        self.CENTER_WALL_X_POSITION = None  # Track center wall x-position (None if not created yet)
        self.WALL_ID = None  # Will be set if walls are added (first outer wall ID for compatibility)
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:e}, km {:e},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        if self.RECORD:
            self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        self.VISION_ATTR = vision_attributes
        #### LiDAR configuration constants (always defined) ####
        # 2D LiDAR constants
        self.LIDAR_MAX_RANGE = 10.0  # Maximum detection range in meters
        self.LIDAR_NUM_RAYS = 360  # Number of rays per scan (angular resolution: 360/num_rays degrees)
        self.LIDAR_FOV = 360.0  # Field of view in degrees (360 = full circle for 2D)
        self.LIDAR_SCAN_RATE_HZ = 10.0  # Desired scan rate in Hz
        self.LIDAR_CAPTURE_FREQ = int(self.CTRL_FREQ/self.LIDAR_SCAN_RATE_HZ)  # Update frequency in control steps
        # 3D LiDAR constants - Polar Range Image Representation
        self.LIDAR3D_MAX_RANGE = 5.0  # Maximum detection range in meters
        self.LIDAR3D_NUM_BEAMS = 16  # Vertical beams (elevation channels) for range image
        self.LIDAR3D_NUM_BINS = 90   # Horizontal bins (azimuth channels) for range image
        self.LIDAR3D_HORIZONTAL_FOV = 360.0  # Horizontal field of view in degrees (full circle)
        self.LIDAR3D_VERTICAL_FOV = 90.0  # Vertical field of view in degrees (hemisphere upward: 0 to +90)
        # Computed resolutions based on fixed beam/bin configuration
        self.LIDAR3D_VERTICAL_RES = self.LIDAR3D_VERTICAL_FOV / (self.LIDAR3D_NUM_BEAMS - 1)  # ~6° per beam
        self.LIDAR3D_HORIZONTAL_RES = self.LIDAR3D_HORIZONTAL_FOV / self.LIDAR3D_NUM_BINS  # 4° per bin
        self.LIDAR3D_SCAN_RATE_HZ = 5.0  # Desired scan rate in Hz (reduced for performance)
        self.LIDAR3D_CAPTURE_FREQ = int(self.CTRL_FREQ/self.LIDAR3D_SCAN_RATE_HZ)  # Update frequency in control steps
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.PYB_STEPS_PER_CTRL != 0:
                print("[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                for i in range(self.NUM_DRONES):
                    os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/"), exist_ok=True)
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.PYB_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES,3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
    
    ################################################################################

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    ################################################################################

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info
    
    ################################################################################
    
    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter*self.PYB_TIMESTEP, self.PYB_FREQ, (self.step_counter*self.PYB_TIMESTEP)/(time.time()-self.RESET_TIME)))
        for i in range (self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0]*self.RAD2DEG, self.rpy[i, 1]*self.RAD2DEG, self.rpy[i, 2]*self.RAD2DEG),
                  "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]))
    
    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)
    
    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS
    
    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])
        #### Remove default damping #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        # for i in range(self.NUM_DRONES):
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()
        if self.CEILING_HEIGHT is not None:
            self._addCeiling()
            # Create 4 outer walls around origin (center wall is added separately during reset)
            self._addOuterWalls()
    
    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range (self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
    
    ################################################################################

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=os.path.join(self.OUTPUT_FOLDER, "video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".mp4"),
                                                physicsClientId=self.CLIENT
                                                )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"), '')
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
    
    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        return state.reshape(20,)

    ################################################################################

    def _getDroneImages(self,
                        nth_drone,
                        segmentation: bool=True
                        ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray 
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat,np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :]+np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0
                                                      )
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.IMG_RES[0],
                                                 height=self.IMG_RES[1],
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    ################################################################################

    def _getDroneLidarScan(self,
                          nth_drone,
                          max_range=None,
                          num_rays=None,
                          fov=None
                          ):
        """Returns a 2D LiDAR scan from the n-th drone's perspective.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        max_range : float, optional
            Maximum detection range in meters. If None, uses self.LIDAR_MAX_RANGE.
        num_rays : int, optional
            Number of rays per scan. If None, uses self.LIDAR_NUM_RAYS.
        fov : float, optional
            Field of view in degrees. If None, uses self.LIDAR_FOV.

        Returns
        -------
        ndarray
            (num_rays,)-shaped array of floats containing the range measurements for each ray.
            Values are in meters, with max_range indicating no hit.
        ndarray
            (num_rays, 3)-shaped array of floats containing the 3D hit points for each ray.
            If no hit, the point is at max_range along the ray direction.
        ndarray
            (num_rays, 2)-shaped array of floats containing the (azimuth, elevation) angles
            in radians for each ray direction.

        """
        #### Use default constants if not specified ################
        if max_range is None:
            max_range = self.LIDAR_MAX_RANGE
        if num_rays is None:
            num_rays = self.LIDAR_NUM_RAYS
        if fov is None:
            fov = self.LIDAR_FOV
        
        #### Generate ray directions in horizontal plane (2D LiDAR) #
        fov_rad = np.deg2rad(fov)
        angles = np.linspace(0, fov_rad, num_rays, endpoint=False)
        
        #### Get drone position and orientation #####################
        drone_pos = self.pos[nth_drone, :]
        drone_quat = self.quat[nth_drone, :]
        rot_mat = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        
        #### Generate ray directions in drone's local frame ##########
        # For 2D LiDAR: rays in horizontal plane (z=0 in local frame)
        ray_dirs_local = np.array([
            [np.cos(angle), np.sin(angle), 0.0] 
            for angle in angles
        ])
        
        #### Transform ray directions to world frame ################
        ray_dirs_world = (rot_mat @ ray_dirs_local.T).T
        
        #### Compute ray start and end positions ####################
        ray_from = np.tile(drone_pos, (num_rays, 1))
        ray_to = ray_from + ray_dirs_world * max_range
        
        #### Perform batch raycast ##################################
        ray_hits = p.rayTestBatch(
            rayFromPositions=ray_from.tolist(),
            rayToPositions=ray_to.tolist(),
            parentObjectUniqueId=self.DRONE_IDS[nth_drone],  # Ignore the drone itself
            physicsClientId=self.CLIENT
        )
        
        #### Extract ranges and hit points ##########################
        ranges = np.zeros(num_rays)
        hit_points = np.zeros((num_rays, 3))
        
        for i, hit in enumerate(ray_hits):
            if hit[0] != -1:  # Hit detected
                ranges[i] = hit[2] * max_range  # hitFraction * max_range
                hit_points[i] = hit[3]  # hitPosition
            else:  # No hit
                ranges[i] = max_range
                hit_points[i] = ray_to[i]
        
        #### Compute world-frame angles from ray directions ##########
        # Calculate azimuth angles in world frame (not local frame)
        # This ensures visualization rotates correctly with the drone
        world_angles = np.arctan2(ray_dirs_world[:, 1], ray_dirs_world[:, 0])
        # Normalize to [0, 2π] range
        world_angles = np.mod(world_angles + 2*np.pi, 2*np.pi)
        ray_angles = np.column_stack([world_angles, np.zeros(num_rays)])  # (azimuth, elevation=0 for 2D)
        
        return ranges, hit_points, ray_angles

    ################################################################################

    def _getDroneLidarScan3D(self,
                             nth_drone,
                             max_range=None,
                             return_point_cloud=False
                             ):
        """Returns a 3D LiDAR scan as a polar range image.
        
        Performs hemispherical (upward) scanning pattern using a fixed-size
        polar range image representation (H x W x 2) where:
        - H = NUM_BEAMS (vertical elevation channels)
        - W = NUM_BINS (horizontal azimuth bins)
        - Channel 0: normalized range [0, 1]
        - Channel 1: hit mask {0, 1}
        
        This representation is designed for CNN-based learning and matches
        realistic rotating multi-beam LiDAR behavior.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        max_range : float, optional
            Maximum detection range in meters. If None, uses self.LIDAR3D_MAX_RANGE.
        return_point_cloud : bool, optional
            If True, also convert and return point cloud for visualization.
            Default is False (only return range image for efficiency).

        Returns
        -------
        range_image : ndarray, shape (H, W, 2), dtype=float32
            Polar range image representation.
            Channel 0: normalized range [0, 1] (distance / max_range)
            Channel 1: hit mask {0, 1} (1 = valid hit, 0 = no return)
        hit_points : ndarray, shape (N, 3) [only if return_point_cloud=True]
            3D hit points in BODY FRAME coordinates (drone at origin).
            Only contains valid hits (where mask == 1).
        ranges : ndarray, shape (N,) [only if return_point_cloud=True]
            Range measurements in meters for valid hits.
        ray_angles : ndarray, shape (N, 2) [only if return_point_cloud=True]
            [azimuth, elevation] angles in radians for valid hits.

        """
        #### Use default constants ################
        if max_range is None:
            max_range = self.LIDAR3D_MAX_RANGE
        
        #### Fixed dimensions for polar range image ########
        num_beams = self.LIDAR3D_NUM_BEAMS  # H = 16 (vertical)
        num_bins = self.LIDAR3D_NUM_BINS    # W = 90 (horizontal)
        
        # Elevation angles: linearly spaced from 0° (horizontal) to 90° (upward)
        elevation_angles = np.linspace(0, np.deg2rad(self.LIDAR3D_VERTICAL_FOV), num_beams, endpoint=True)
        
        # Azimuth angles: 0° to 360° (exclusive), centered on bin
        azimuth_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
        
        # Create meshgrid for all beam-bin combinations: (num_beams, num_bins)
        az_grid, el_grid = np.meshgrid(azimuth_angles, elevation_angles, indexing='xy')
        az_flat = az_grid.flatten()
        el_flat = el_grid.flatten()
        
        #### Get drone position and orientation #####################
        drone_pos = np.array(self.pos[nth_drone, :])
        drone_quat = self.quat[nth_drone, :]
        rot_mat = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        
        #### Calculate LiDAR origin offset to avoid drone body interference ####
        # Position LiDAR on top of the drone body to avoid self-collision
        # Offset = half collision height + z offset + small margin (0.05m)
        lidar_z_offset = self.COLLISION_H / 2 + self.COLLISION_Z_OFFSET + 0.05
        # LiDAR origin in drone's body frame (at top of drone, centered)
        lidar_origin_body = np.array([0, 0, lidar_z_offset])
        # Transform to world frame
        lidar_origin_world = drone_pos + rot_mat @ lidar_origin_body
        
        #### Generate ray directions in drone's local frame (vectorized) ####
        # Convert spherical to Cartesian coordinates
        # In drone's local frame: X=forward, Y=left, Z=up
        cos_el = np.cos(el_flat)
        sin_el = np.sin(el_flat)
        cos_az = np.cos(az_flat)
        sin_az = np.sin(az_flat)
        
        ray_dirs_local = np.column_stack([
            cos_el * cos_az,  # X: Forward component
            cos_el * sin_az,  # Y: Left component
            sin_el            # Z: Upward component (positive for upward)
        ])
        
        #### Apply LiDAR mount pitch rotation (10° forward) ####
        # Rotate all ray directions by +10° around Y axis (pitch forward)
        # This simulates the LiDAR being mounted at a 10° forward angle
        lidar_pitch_deg = 10.0  # Forward pitch angle in degrees
        lidar_pitch_rad = np.deg2rad(lidar_pitch_deg)
        cos_pitch = np.cos(lidar_pitch_rad)
        sin_pitch = np.sin(lidar_pitch_rad)
        
        # Rotation matrix around Y axis (pitch forward)
        # Standard right-hand rule: positive rotation rotates Z toward +X (forward tilt)
        pitch_rotation = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        # Apply pitch rotation to ray directions in body frame
        ray_dirs_local = (pitch_rotation @ ray_dirs_local.T).T
        
        #### Transform ray directions to world frame ################
        ray_dirs_world = (rot_mat @ ray_dirs_local.T).T
        
        #### Compute ray start and end positions ####################
        total_rays = num_beams * num_bins
        
        # Rays start from the LiDAR origin (on top of drone), not from drone center
        ray_from = np.tile(lidar_origin_world, (total_rays, 1))
        ray_to = ray_from + ray_dirs_world * max_range
        
        #### Perform batch raycast ####
        # Update collision detection before raycasting
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        
        # Raycast returns closest hit per ray
        ray_hits = p.rayTestBatch(
            rayFromPositions=ray_from.tolist(),
            rayToPositions=ray_to.tolist(),
            parentObjectUniqueId=-1,  # Test all objects
            physicsClientId=self.CLIENT
        )
        
        #### Build range image from raycast results ####
        # Initialize range image: (H, W, 2)
        # Channel 0: normalized range [0, 1]
        # Channel 1: hit mask {0, 1}
        range_image = np.ones((num_beams, num_bins, 2), dtype=np.float32)
        range_image[:, :, 1] = 0.0  # Initialize all masks to 0 (no hit)
        
        for i, hit in enumerate(ray_hits):
            # Compute elevation and azimuth indices from flattened index
            # Rows are elevation (beams), columns are azimuth (bins)
            e = i // num_bins  # Elevation index (0 to num_beams-1)
            a = i % num_bins   # Azimuth index (0 to num_bins-1)
            
            if hit[0] != -1:  # Hit detected
                # hit[0] = objectUniqueId
                # hit[2] = hitFraction (0.0 = at ray start, 1.0 = at ray end)
                # Filter self-hits (drone body collisions)
                if hit[0] == int(self.DRONE_IDS[nth_drone]):
                    # Self-hit: treat as no return
                    range_image[e, a, 0] = 1.0  # Max range (normalized)
                    range_image[e, a, 1] = 0.0  # No hit mask
                else:
                    # Valid hit: store normalized range and set mask
                    hit_distance = hit[2] * max_range
                    range_image[e, a, 0] = min(hit_distance / max_range, 1.0)  # Normalized range [0, 1]
                    range_image[e, a, 1] = 1.0  # Hit mask
            else:  # No hit
                range_image[e, a, 0] = 1.0  # Max range (normalized)
                range_image[e, a, 1] = 0.0  # No hit mask
        
        #### Return range image, optionally with point cloud ####
        if return_point_cloud:
            # Convert range image to point cloud for visualization
            hit_points, ranges, ray_angles = self._range_image_to_point_cloud(
                range_image, elevation_angles, azimuth_angles, max_range,
                pitch_rotation
            )
            return range_image, hit_points, ranges, ray_angles
        else:
            # Return only range image (efficient for learning)
            return range_image

    ################################################################################

    def _range_image_to_point_cloud(self, range_image, elevation_angles, azimuth_angles,
                                     max_range, pitch_rotation):
        """Convert polar range image to point cloud (body frame).
        
        This is a helper method for visualization ONLY. Learning pipelines should
        operate directly on the range image without conversion to point cloud.
        
        Parameters
        ----------
        range_image : ndarray, shape (H, W, 2)
            Polar range image with channels [range, hit_mask].
            Range is normalized [0, 1], mask is {0, 1}.
        elevation_angles : ndarray, shape (H,)
            Elevation angles in radians (0 = horizontal, +90° = upward).
        azimuth_angles : ndarray, shape (W,)
            Azimuth angles in radians (0 to 2π).
        max_range : float
            Maximum range in meters (for denormalization).
        pitch_rotation : ndarray, shape (3, 3)
            LiDAR pitch rotation matrix (10° forward).
        
        Returns
        -------
        hit_points : ndarray, shape (N, 3)
            3D hit points in body frame (only valid hits where mask == 1).
            Coordinates: X=forward, Y=left, Z=up.
        ranges : ndarray, shape (N,)
            Range measurements in meters (only valid hits).
        ray_angles : ndarray, shape (N, 2)
            [azimuth, elevation] in radians for each valid hit.
        """
        H, W = range_image.shape[:2]
        
        # Extract ranges and masks
        ranges_normalized = range_image[:, :, 0]  # (H, W)
        hit_masks = range_image[:, :, 1]          # (H, W)
        
        # Find valid hits (mask > 0.5 to handle float precision)
        valid_mask = hit_masks > 0.5
        valid_indices = np.where(valid_mask)
        
        # Get elevation and azimuth for valid hits
        el_valid = elevation_angles[valid_indices[0]]
        az_valid = azimuth_angles[valid_indices[1]]
        ranges_valid = ranges_normalized[valid_mask] * max_range
        
        # Convert spherical to Cartesian (body frame, before pitch)
        # In body frame: X=forward, Y=left, Z=up
        cos_el = np.cos(el_valid)
        sin_el = np.sin(el_valid)
        cos_az = np.cos(az_valid)
        sin_az = np.sin(az_valid)
        
        hit_points_local = np.column_stack([
            ranges_valid * cos_el * cos_az,  # X: forward
            ranges_valid * cos_el * sin_az,  # Y: left
            ranges_valid * sin_el             # Z: up
        ])
        
        # Apply pitch rotation (10° forward)
        hit_points_local = (pitch_rotation @ hit_points_local.T).T
        
        # Prepare ray angles output
        ray_angles = np.column_stack([az_valid, el_valid])
        
        return hit_points_local, ranges_valid, ray_angles

    ################################################################################

    def _exportImage(self,
                     img_type: ImageType,
                     img_input,
                     path: str,
                     frame_num: int=0
                     ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype('uint8'), 'RGBA')).save(os.path.join(path,"frame_"+str(frame_num)+".png"))
        elif img_type == ImageType.DEP:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.SEG:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype('uint8')
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(os.path.join(path,"frame_"+str(frame_num)+".png"))

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix 
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES-1):
            for j in range(self.NUM_DRONES-i-1):
                if np.linalg.norm(self.pos[i, :]-self.pos[j+i+1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i, j+i+1] = adjacency_mat[j+i+1, i] = 1
        return adjacency_mat
    
    ################################################################################
    
    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )

    ################################################################################

    def _groundEffect(self,
                      rpm,
                      nth_drone
                      ):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Kin. info of all links (propellers and center of mass)
        link_states = p.getLinkStates(self.DRONE_IDS[nth_drone],
                                        linkIndices=[0, 1, 2, 3, 4],
                                        computeLinkVelocity=1,
                                        computeForwardKinematics=1,
                                        physicsClientId=self.CLIENT
                                        )
        #### Simple, per-propeller ground effects ##################
        prop_heights = np.array([link_states[0][0][2], link_states[1][0][2], link_states[2][0][2], link_states[3][0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS/(4 * prop_heights))**2
        if np.abs(self.rpy[nth_drone,0]) < np.pi/2 and np.abs(self.rpy[nth_drone,1]) < np.pi/2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )
    
    ################################################################################

    def _drag(self,
              rpm,
              nth_drone
              ):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot.T, drag_factors*np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )
    
    ################################################################################

    def _downwash(self,
                  nth_drone
                  ):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10: # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS/(4*delta_z))**2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5*(delta_xy/beta)**2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )

    ################################################################################

    def _dynamics(self,
                  rpm,
                  nth_drone
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone,:]
        quat = self.quat[nth_drone,:]
        vel = self.vel[nth_drone,:]
        rpy_rates = self.rpy_rates[nth_drone,:]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2)*self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            z_torques = -z_torques
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL==DroneModel.RACE:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2X:
            x_torque = - (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
        pos = pos + self.PYB_TIMESTEP * vel
        quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          quat,
                                          physicsClientId=self.CLIENT
                                          )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            np.dot(rotation, rpy_rates),
                            physicsClientId=self.CLIENT
                            )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone,:] = rpy_rates

    def _integrateQ(self, quat, omega, dt):
        omega_norm = np.linalg.norm(omega)
        p, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat
        lambda_ = np.array([
            [ 0,  r, -q, p],
            [-r,  0,  p, q],
            [ q, -p,  0, r],
            [-p, -q, -r, 0]
        ]) * .5
        theta = omega_norm * dt / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat

    ################################################################################

    def _normalizedActionToRPM(self,
                               action
                               ):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action) > 1):
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action+1)*self.HOVER_RPM, self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM)*action) # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`
    
    ################################################################################

    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2*self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Y_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Z_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
    
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.
        
        Note: Sphere and cube have been removed - only wall and ceiling remain.

        """
        # Obstacles removed - only wall and ceiling are used
        pass
    
    ################################################################################
    
    def _addCeiling(self):
        """Add a ceiling at the specified height using tiled collision shapes.

        Uses 5m×5m tiles to avoid PyBullet's raycast issues with very large shapes.
        Ceiling covers the entire 15m × 15m room bounded by the 4 outer walls, without extending past them.

        """
        ceiling_thickness = 0.3  # Thickness of ceiling tiles
        tile_size = 5.0  # Each tile is 5m × 5m (optimal for PyBullet)
        room_size = self.ROOM_SIZE  # 15m × 15m room
        
        # Ceiling covers the entire room: x and y from -7.5 to +7.5
        ceiling_start = -room_size / 2
        ceiling_end = room_size / 2
        
        # Calculate number of tiles needed (15m / 5m = 3 tiles per side)
        num_tiles = max(1, int(np.ceil(room_size / tile_size)))
        
        tile_half_extents = [tile_size / 2, tile_size / 2, ceiling_thickness / 2]
        
        # Store all ceiling tile IDs
        self.CEILING_TILE_IDS = []
        
        # Create tiles in a grid covering the entire room
        for ix in range(num_tiles):
            for iy in range(num_tiles):
                x_pos = ceiling_start + tile_size / 2 + ix * tile_size
                y_pos = ceiling_start + tile_size / 2 + iy * tile_size
                z_pos = self.CEILING_HEIGHT + ceiling_thickness / 2
                
                tile_collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=tile_half_extents,
                    physicsClientId=self.CLIENT
                )
                tile_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=tile_half_extents,
                    rgbaColor=[0.8, 0.8, 0.8, 1.0],
                    physicsClientId=self.CLIENT
                )
                tile_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=tile_collision,
                    baseVisualShapeIndex=tile_visual,
                    basePosition=[x_pos, y_pos, z_pos],
                    physicsClientId=self.CLIENT
                )
                self.CEILING_TILE_IDS.append(tile_id)
        
        # Use the first tile as the main CEILING_ID for compatibility
        self.CEILING_ID = self.CEILING_TILE_IDS[0] if self.CEILING_TILE_IDS else None
        
        # Step simulation to ensure collision shapes are initialized
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.CLIENT)
    
    ################################################################################
    
    def _addOuterWalls(self):
        """Add 4 outer walls around the origin, each 15m long, forming a square room.

        Uses 5m-wide vertical cubes to avoid PyBullet's raycast issues with very large shapes.
        Creates 4 walls: North (y=+7.5), South (y=-7.5), East (x=+7.5), West (x=-7.5).

        """
        wall_height = self.CEILING_HEIGHT if self.CEILING_HEIGHT is not None else 10.0
        room_size = self.ROOM_SIZE  # 15m × 15m room
        wall_position = room_size / 2  # Walls at ±7.5m
        
        # Create wall from 5m-wide cubes (optimal for PyBullet)
        cube_length = 5.0  # Each cube segment is 5m long
        cube_height = wall_height  # Full height (avoids vertical seams)
        wall_thickness = 0.5  # Wall thickness
        
        # Calculate number of cubes needed per wall (15m / 5m = 3 cubes per wall)
        num_cubes_per_wall = max(1, int(np.ceil(room_size / cube_length)))
        
        # Store outer wall cube IDs
        self.WALL_CUBE_IDS = []
        
        # Create 4 outer walls: North, South, East, West
        wall_configs = [
            # (name, axis, position, orientation)
            ("North", "x", wall_position, 0),   # Wall at y=+7.5, extends in x-direction
            ("South", "x", -wall_position, 0),  # Wall at y=-7.5, extends in x-direction
            ("East", "y", wall_position, 90),   # Wall at x=+7.5, extends in y-direction
            ("West", "y", -wall_position, 90),  # Wall at x=-7.5, extends in y-direction
        ]
        
        for wall_name, axis, position, rotation_deg in wall_configs:
            for i in range(num_cubes_per_wall):
                # Calculate position along the wall
                offset = -room_size / 2 + cube_length / 2 + i * cube_length
                
                if axis == "x":
                    # Wall extends in x-direction (North/South walls)
                    x_pos = offset
                    y_pos = position
                    # Half extents: [length/2, thickness/2, height/2]
                    cube_half_extents = [cube_length / 2, wall_thickness / 2, cube_height / 2]
                else:  # axis == "y"
                    # Wall extends in y-direction (East/West walls)
                    x_pos = position
                    y_pos = offset
                    # Half extents: [thickness/2, length/2, height/2]
                    cube_half_extents = [wall_thickness / 2, cube_length / 2, cube_height / 2]
                
                z_pos = cube_height / 2  # Center vertically
                
                cube_collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=cube_half_extents,
                    physicsClientId=self.CLIENT
                )
                cube_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=cube_half_extents,
                    rgbaColor=[0.7, 0.7, 0.7, 1.0],
                    physicsClientId=self.CLIENT
                )
                cube_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=cube_collision,
                    baseVisualShapeIndex=cube_visual,
                    basePosition=[x_pos, y_pos, z_pos],
                    physicsClientId=self.CLIENT
                )
                self.WALL_CUBE_IDS.append(cube_id)
        
        # Use the first cube as the main WALL_ID for compatibility
        self.WALL_ID = self.WALL_CUBE_IDS[0] if self.WALL_CUBE_IDS else None
        
        # Step simulation to ensure collision shapes are initialized
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.CLIENT)
    
    def _addCenterWall(self, x_position: float = 0.0, window_position: list = None):
        """Add center wall that splits the room into two halves (at specified x position, extends in y-direction).

        Uses 5m-wide vertical cubes to avoid PyBullet's raycast issues with very large shapes.
        Currently fixed at x=0 by default, but can be modified in the future for randomization.

        Parameters:
            x_position : float, optional
                - X position of the center wall. Default is 0.0.
            window_position : list, optional
                - [y_center, z_center] position of the window center in meters. Default is None.
                - The window position is stored for task logic but doesn't create a physical hole in the wall.
        """
        wall_height = self.CEILING_HEIGHT if self.CEILING_HEIGHT is not None else 10.0
        room_size = self.ROOM_SIZE  # 15m × 15m room
        
        # Create wall from 5m-wide cubes (optimal for PyBullet)
        cube_length = 5.0  # Each cube segment is 5m long
        cube_height = wall_height  # Full height (avoids vertical seams)
        wall_thickness = 0.5  # Wall thickness
        
        # Calculate number of cubes needed per wall (15m / 5m = 3 cubes per wall)
        num_cubes_per_wall = max(1, int(np.ceil(room_size / cube_length)))
        
        # Store center wall cube IDs
        self.CENTER_WALL_CUBE_IDS = []
        
        # Add center wall that splits the room into two halves (at specified x position, extends in y-direction)
        center_wall_position = x_position
        for i in range(num_cubes_per_wall):
            # Calculate position along the wall
            offset = -room_size / 2 + cube_length / 2 + i * cube_length
            x_pos = center_wall_position
            y_pos = offset
            # Half extents: [thickness/2, length/2, height/2]
            cube_half_extents = [wall_thickness / 2, cube_length / 2, cube_height / 2]
            z_pos = cube_height / 2  # Center vertically
            
            cube_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=cube_half_extents,
                physicsClientId=self.CLIENT
            )
            cube_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=cube_half_extents,
                rgbaColor=[0.7, 0.7, 0.7, 1.0],
                physicsClientId=self.CLIENT
            )
            cube_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=cube_collision,
                baseVisualShapeIndex=cube_visual,
                basePosition=[x_pos, y_pos, z_pos],
                physicsClientId=self.CLIENT
            )
            self.CENTER_WALL_CUBE_IDS.append(cube_id)
        
        # Track the wall position
        self.CENTER_WALL_X_POSITION = center_wall_position
        
        # Store window position if provided (for task logic, wall remains solid)
        if window_position is not None:
            self.CENTER_WALL_WINDOW_POSITION = window_position
        
        # Step simulation to ensure collision shapes are initialized
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.CLIENT)
    
    def _removeCenterWall(self):
        """Remove the center wall cubes from the simulation."""
        import pybullet as p
        for cube_id in self.CENTER_WALL_CUBE_IDS:
            p.removeBody(cube_id, physicsClientId=self.CLIENT)
        self.CENTER_WALL_CUBE_IDS = []
        self.CENTER_WALL_X_POSITION = None
    
    ################################################################################
    
    def _addVerticalPoles(self, pole_positions, pole_diameter, pole_height=None):
        """Add vertical cylindrical poles from floor to ceiling.
        
        Parameters:
            pole_positions : list of [x, y] pairs
                - List of [x, y] positions for each pole in meters.
            pole_diameter : float
                - Diameter of the poles in meters.
            pole_height : float, optional
                - Height of the poles in meters. If None, uses CEILING_HEIGHT.
        """
        if pole_height is None:
            pole_height = self.CEILING_HEIGHT if self.CEILING_HEIGHT is not None else 10.0
        
        pole_radius = pole_diameter / 2.0
        pole_z_center = pole_height / 2.0  # Center vertically
        
        # Store pole IDs
        self.POLE_IDS = []
        
        for x_pos, y_pos in pole_positions:
            # Create cylindrical collision shape
            pole_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=pole_radius,
                height=pole_height,
                physicsClientId=self.CLIENT
            )
            # Create cylindrical visual shape
            pole_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=pole_radius,
                length=pole_height,
                rgbaColor=[0.6, 0.6, 0.6, 1.0],  # Gray color
                physicsClientId=self.CLIENT
            )
            # Create the pole body
            pole_id = p.createMultiBody(
                baseMass=0,  # Static object
                baseCollisionShapeIndex=pole_collision,
                baseVisualShapeIndex=pole_visual,
                basePosition=[x_pos, y_pos, pole_z_center],
                physicsClientId=self.CLIENT
            )
            self.POLE_IDS.append(pole_id)
        
        # Step simulation to ensure collision shapes are initialized
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.CLIENT)
    
    ################################################################################
    
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _calculateNextStep(self, current_position, destination, step_size=1):
        """
        Calculates intermediate waypoint
        towards drone's destination
        from drone's current position

        Enables drones to reach distant waypoints without
        losing control/crashing, and hover on arrival at destintion

        Parameters
        ----------
        current_position : ndarray
            drone's current position from state vector
        destination : ndarray
            drone's target position 
        step_size: int
            distance next waypoint is from current position, default 1

        Returns
        ----------
        next_pos: int 
            intermediate waypoint for drone

        """
        direction = (
            destination - current_position
        )  # Calculate the direction vector
        distance = np.linalg.norm(
            direction
        )  # Calculate the distance to the destination

        if distance <= step_size:
            # If the remaining distance is less than or equal to the step size,
            # return the destination
            return destination

        normalized_direction = (
            direction / distance
        )  # Normalize the direction vector
        next_step = (
            current_position + normalized_direction * step_size
        )  # Calculate the next step
        return next_step
