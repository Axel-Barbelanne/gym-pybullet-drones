"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SHOW_CAMERA = True
DEFAULT_CAMERA_UPDATE_FREQ = 5  # Update camera every N control steps
DEFAULT_SHOW_LIDAR = True
DEFAULT_LIDAR_UPDATE_FREQ = 5  # Update LiDAR every N control steps
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        show_camera=DEFAULT_SHOW_CAMERA,
        camera_update_freq=DEFAULT_CAMERA_UPDATE_FREQ,
        show_lidar=DEFAULT_SHOW_LIDAR,
        lidar_update_freq=DEFAULT_LIDAR_UPDATE_FREQ
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### Initialize trajectory ####################################
    PERIOD = 10
    # For continuous trajectories (helix, spiral), use longer period to avoid reset
    USE_CONTINUOUS_TRAJECTORY = True  # Set to True for helix/spiral, False for closed loops
    if USE_CONTINUOUS_TRAJECTORY:
        PERIOD = duration_sec  # Use full simulation duration for continuous trajectories
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    TARGET_RPY = np.zeros((NUM_WP,3))  # Store target orientation (roll, pitch, yaw) for each waypoint
    
    #### OPTION 1: Circular trajectory (default) ##################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP)*(2*np.pi) + np.pi/2  # Angle parameter
    #     x = R*np.cos(t) + INIT_XYZS[0, 0]
    #     y = R*np.sin(t) - R + INIT_XYZS[0, 1]
    #     z = 0
    #     TARGET_POS[i, :] = x, y, z
    #     
    #     # Calculate yaw angle to face tangent to the circle (direction of motion)
    #     # For a circle: x = R*cos(t), y = R*sin(t)
    #     # Tangent vector: dx/dt = -R*sin(t), dy/dt = R*cos(t)
    #     # Yaw = atan2(dy/dt, dx/dt) = atan2(R*cos(t), -R*sin(t)) = atan2(cos(t), -sin(t))
    #     # This simplifies to: yaw = t + pi/2 (for counterclockwise motion)
    #     yaw = t + np.pi/2
    #     TARGET_RPY[i, :] = [INIT_RPYS[0, 0], INIT_RPYS[0, 1], yaw]  # Keep roll/pitch, update yaw
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    
    #### OPTION 2: Figure-8 (Lemniscate) trajectory ###############
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 2 * np.pi
    #     # Lemniscate of Bernoulli (figure-8)
    #     scale = 0.4
    #     x = scale * np.sin(t) / (1 + np.cos(t)**2) + INIT_XYZS[0, 0]
    #     y = scale * np.sin(t) * np.cos(t) / (1 + np.cos(t)**2) - R + INIT_XYZS[0, 1]
    #     TARGET_POS[i, :] = x, y, 0
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    
    #### OPTION 3: Square path #####################################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 4  # 4 sides
    #     side = int(t) % 4
    #     progress = t - int(t)
    #     size = 0.5
    #     center_x, center_y = INIT_XYZS[0, 0], INIT_XYZS[0, 1] - R
    #     if side == 0:  # Right side
    #         x = center_x + size
    #         y = center_y - size + 2*size*progress
    #     elif side == 1:  # Top side
    #         x = center_x + size - 2*size*progress
    #         y = center_y + size
    #     elif side == 2:  # Left side
    #         x = center_x - size
    #         y = center_y + size - 2*size*progress
    #     else:  # Bottom side
    #         x = center_x - size + 2*size*progress
    #         y = center_y - size
    #     TARGET_POS[i, :] = x, y, 0
    # wp_counters = np.array([int((i*NUM_WP/4)%NUM_WP) for i in range(num_drones)])
    
    #### OPTION 4: Spiral trajectory (expanding) ###################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 4 * np.pi  # 2 full rotations
    #     r = 0.1 + 0.3 * (i/NUM_WP)  # Radius increases from 0.1 to 0.4
    #     x = r * np.cos(t) + INIT_XYZS[0, 0]
    #     y = r * np.sin(t) - R + INIT_XYZS[0, 1]
    #     TARGET_POS[i, :] = x, y, 0
    # wp_counters = np.array([0 for i in range(num_drones)])
    
    #### OPTION 5: Straight line with 90-degree turns ##############
    # for i in range(NUM_WP):
    #     segment = int((i/NUM_WP) * 4) % 4
    #     progress = ((i/NUM_WP) * 4) % 1.0
    #     length = 0.6
    #     center_x, center_y = INIT_XYZS[0, 0], INIT_XYZS[0, 1] - R
    #     if segment == 0:  # Move forward (positive Y)
    #         x = center_x
    #         y = center_y + length * progress
    #     elif segment == 1:  # Move right (positive X)
    #         x = center_x + length * progress
    #         y = center_y + length
    #     elif segment == 2:  # Move back (negative Y)
    #         x = center_x + length
    #         y = center_y + length - length * progress
    #     else:  # Move left (negative X)
    #         x = center_x + length - length * progress
    #         y = center_y
    #     TARGET_POS[i, :] = x, y, 0
    # wp_counters = np.array([0 for i in range(num_drones)])
    
    #### OPTION 6: Zigzag pattern ##################################
    # num_segments = 8
    # for i in range(NUM_WP):
    #     segment = int((i/NUM_WP) * num_segments) % num_segments
    #     progress = ((i/NUM_WP) * num_segments) % 1.0
    #     x_range = 0.8
    #     y_range = 0.6
    #     center_x, center_y = INIT_XYZS[0, 0], INIT_XYZS[0, 1] - R
    #     x = center_x - x_range/2 + (x_range * progress)
    #     y = center_y - y_range/2 + (y_range * (segment / num_segments)) + (y_range/num_segments if segment % 2 == 1 else 0)
    #     TARGET_POS[i, :] = x, y, 0
    # wp_counters = np.array([0 for i in range(num_drones)])
    
    #### OPTION 7: Infinity symbol (horizontal figure-8) ###########
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 2 * np.pi
    #     scale = 0.4
    #     x = scale * np.sin(t) + INIT_XYZS[0, 0]
    #     y = scale * np.sin(t) * np.cos(t) - R + INIT_XYZS[0, 1]
    #     TARGET_POS[i, :] = x, y, 0
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    
    #### OPTION 8: Diamond path ####################################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 4  # 4 sides
    #     side = int(t) % 4
    #     progress = t - int(t)
    #     size = 0.5
    #     center_x, center_y = INIT_XYZS[0, 0], INIT_XYZS[0, 1] - R
    #     if side == 0:  # Top-right
    #         x = center_x + size * progress
    #         y = center_y + size * progress
    #     elif side == 1:  # Top-left
    #         x = center_x + size - size * progress
    #         y = center_y + size + size * progress
    #     elif side == 2:  # Bottom-left
    #         x = center_x - size * progress
    #         y = center_y + size - size * progress
    #     else:  # Bottom-right
    #         x = center_x - size + size * progress
    #         y = center_y - size * progress
    #     TARGET_POS[i, :] = x, y, 0
    # wp_counters = np.array([int((i*NUM_WP/4)%NUM_WP) for i in range(num_drones)])
    
    #### OPTION 9: Helix (3D spiral) - Continuous ##################
    # This trajectory continues indefinitely without resetting
    # Set USE_CONTINUOUS_TRAJECTORY = True above for seamless continuation
    for i in range(NUM_WP):
        t = (i/NUM_WP) * 4 * np.pi  # 2 full rotations per period
        r = 0.3  # Constant radius
        x = r * np.cos(t) + INIT_XYZS[0, 0]
        y = r * np.sin(t) - R + INIT_XYZS[0, 1]
        z = INIT_XYZS[0, 2] + 0.3 * (i/NUM_WP)  # Ascends from initial height
        TARGET_POS[i, :] = x, y, z
        
        # Calculate yaw angle to face tangent to the 2D circle (direction of motion in XY plane)
        # For the helix, the 2D projection is a circle: x = r*cos(t), y = r*sin(t)
        # Tangent vector in XY plane: dx/dt = -r*sin(t), dy/dt = r*cos(t)
        # Yaw = atan2(dy/dt, dx/dt) = atan2(r*cos(t), -r*sin(t)) = atan2(cos(t), -sin(t))
        # This simplifies to: yaw = t + pi/2 (for counterclockwise motion)
        # Note: This aligns yaw with 2D motion direction, which is natural for a helix
        yaw = t + np.pi/2
        TARGET_RPY[i, :] = [INIT_RPYS[0, 0], INIT_RPYS[0, 1], yaw]  # Keep roll/pitch, update yaw
    wp_counters = np.array([0 for i in range(num_drones)])
    
    #### OPTION 10: Circular with altitude waves ###################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 2 * np.pi
    #     r = 0.3
    #     x = r * np.cos(t + np.pi/2) + INIT_XYZS[0, 0]
    #     y = r * np.sin(t + np.pi/2) - R + INIT_XYZS[0, 1]
    #     z = INIT_XYZS[0, 2] + 0.2 * np.sin(3 * t)  # 3 altitude waves
    #     TARGET_POS[i, :] = x, y, z
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### OPTION 11: Square path with ascending altitude ############
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 4  # 4 sides
    #     side = int(t) % 4
    #     progress = t - int(t)
    #     size = 0.5
    #     center_x, center_y = INIT_XYZS[0, 0], INIT_XYZS[0, 1] - R
    #     if side == 0:  # Right side
    #         x = center_x + size
    #         y = center_y - size + 2*size*progress
    #     elif side == 1:  # Top side
    #         x = center_x + size - 2*size*progress
    #         y = center_y + size
    #     elif side == 2:  # Left side
    #         x = center_x - size
    #         y = center_y + size - 2*size*progress
    #     else:  # Bottom side
    #         x = center_x - size + 2*size*progress
    #         y = center_y - size
    #     z = INIT_XYZS[0, 2] + 0.4 * (i/NUM_WP)  # Gradually ascends
    #     TARGET_POS[i, :] = x, y, z
    # wp_counters = np.array([int((i*NUM_WP/4)%NUM_WP) for i in range(num_drones)])
    
    #### OPTION 12: Figure-8 with vertical loops ###################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 2 * np.pi
    #     scale = 0.4
    #     x = scale * np.sin(t) / (1 + np.cos(t)**2) + INIT_XYZS[0, 0]
    #     y = scale * np.sin(t) * np.cos(t) / (1 + np.cos(t)**2) - R + INIT_XYZS[0, 1]
    #     z = INIT_XYZS[0, 2] + 0.15 * np.sin(2 * t)  # Vertical loops
    #     TARGET_POS[i, :] = x, y, z
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    
    #### OPTION 13: Spiral up and down #############################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 6 * np.pi  # 3 full rotations
    #     r = 0.3 * (1 - abs((i/NUM_WP) * 2 - 1))  # Radius: 0.3 -> 0 -> 0.3
    #     x = r * np.cos(t) + INIT_XYZS[0, 0]
    #     y = r * np.sin(t) - R + INIT_XYZS[0, 1]
    #     z = INIT_XYZS[0, 2] + 0.4 * np.sin(np.pi * i/NUM_WP)  # Up then down
    #     TARGET_POS[i, :] = x, y, z
    # wp_counters = np.array([0 for i in range(num_drones)])
    
    #### OPTION 14: Vertical corkscrew ##############################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 4 * np.pi  # 2 full rotations
    #     r = 0.25
    #     x = r * np.cos(t) + INIT_XYZS[0, 0]
    #     y = r * np.sin(t) - R + INIT_XYZS[0, 1]
    #     z = INIT_XYZS[0, 2] + 0.3 * (i/NUM_WP) + 0.1 * np.sin(4 * t)  # Ascends with waves
    #     TARGET_POS[i, :] = x, y, z
    # wp_counters = np.array([0 for i in range(num_drones)])
    
    #### OPTION 15: 3D Lissajous curve ##############################
    # for i in range(NUM_WP):
    #     t = (i/NUM_WP) * 2 * np.pi
    #     scale = 0.4
    #     x = scale * np.sin(2 * t) + INIT_XYZS[0, 0]
    #     y = scale * np.sin(3 * t) - R + INIT_XYZS[0, 1]
    #     z = INIT_XYZS[0, 2] + 0.2 * np.sin(5 * t)  # Z oscillates
    #     TARGET_POS[i, :] = x, y, z
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    
    #### OPTION 16: Staircase pattern ##############################
    # num_steps = 5
    # step_height = 0.15
    # step_size = 0.4
    # for i in range(NUM_WP):
    #     step = int((i/NUM_WP) * num_steps) % num_steps
    #     progress = ((i/NUM_WP) * num_steps) % 1.0
    #     center_x, center_y = INIT_XYZS[0, 0], INIT_XYZS[0, 1] - R
    #     # Move in a square pattern, ascending each step
    #     segment = int(progress * 4) % 4
    #     seg_progress = (progress * 4) % 1.0
    #     if segment == 0:  # Forward
    #         x = center_x
    #         y = center_y + step_size * seg_progress
    #     elif segment == 1:  # Right
    #         x = center_x + step_size * seg_progress
    #         y = center_y + step_size
    #     elif segment == 2:  # Back
    #         x = center_x + step_size
    #         y = center_y + step_size - step_size * seg_progress
    #     else:  # Left
    #         x = center_x + step_size - step_size * seg_progress
    #         y = center_y
    #     z = INIT_XYZS[0, 2] + step * step_height
    #     TARGET_POS[i, :] = x, y, z
    # wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui,
                        vision_attributes=show_camera  # Enable camera if show_camera is True
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    
    #### Position PyBullet GUI window (vertically centered) #####
    if gui:
        try:
            import subprocess
            import time
            import tkinter as tk
            # Get actual screen dimensions
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            
            # Wait a moment for PyBullet window to be created
            time.sleep(0.8)  # Increased wait time for window to fully initialize
            
            # Calculate vertical center position (assuming window height ~800px)
            window_height = 800
            y_center = (screen_height - window_height) // 2
            
            # Try multiple methods to position the window
            success = False
            
            # Method 1: Try wmctrl (Linux)
            try:
                result = subprocess.run(['wmctrl', '-r', 'PyBullet', '-e', f'0,50,{y_center},800,800'], 
                                     capture_output=True, timeout=1, check=False)
                if result.returncode == 0:
                    success = True
            except:
                pass
            
            # Method 2: Try xdotool (Linux)
            if not success:
                try:
                    result = subprocess.run(['xdotool', 'search', '--name', 'PyBullet', 'windowmove', '50', str(y_center)],
                                         capture_output=True, timeout=1, check=False)
                    if result.returncode == 0:
                        success = True
                except:
                    pass
            
            # Method 3: Try xdotool with class name (alternative)
            if not success:
                try:
                    result = subprocess.run(['xdotool', 'search', '--class', 'bullet', 'windowmove', '50', str(y_center)],
                                         capture_output=True, timeout=1, check=False)
                except:
                    pass
        except Exception as e:
            # If all methods fail, print debug info (only once)
            if gui:
                print(f"[INFO] Could not position PyBullet window automatically: {e}")
                print(f"[INFO] Screen dimensions: {screen_width}x{screen_height}, target y: {y_center}")

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    
    #### Define window sizes (shared constants) ##################
    LIDAR_WINDOW_WIDTH = 500  # Width in pixels (will match camera)
    CAMERA_WINDOW_WIDTH = LIDAR_WINDOW_WIDTH  # Exact match with LiDAR
    CAMERA_WINDOW_HEIGHT = 500  # Square window
    
    #### Initialize LiDAR visualization #######################
    lidar_fig = None
    lidar_ax = None
    if show_lidar:
        plt.ion()  # Turn on interactive mode
        # Create smaller figure for top-right positioning
        # Calculate figure size to match desired pixel width (assuming ~100 DPI)
        lidar_fig = plt.figure(figsize=(LIDAR_WINDOW_WIDTH/100, LIDAR_WINDOW_WIDTH/100))
        lidar_ax = lidar_fig.add_subplot(111, projection='polar')
        # Rotate plot so 0° (forward) appears at the top instead of right
        lidar_ax.set_theta_offset(np.pi/2)  # Rotate by 90° clockwise
        lidar_ax.set_theta_direction(-1)  # Reverse direction so angles increase clockwise
        # Position window in top-right corner
        try:
            mgr = lidar_fig.canvas.manager
            if hasattr(mgr, 'window'):
                # Position at top-right: x=1200 (right side), y=50 (top)
                mgr.window.wm_geometry("+1200+50")
        except:
            pass  # Fallback if positioning fails

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    
    #### Initialize camera window positioning ###################
    if show_camera:
        # Create named window for positioning control
        cv2.namedWindow('Drone Camera Feed', cv2.WINDOW_NORMAL)
        # Make camera window exactly same width as LiDAR
        cv2.resizeWindow('Drone Camera Feed', int(CAMERA_WINDOW_WIDTH), int(CAMERA_WINDOW_HEIGHT))
        # Position window in bottom-right corner (lower than before)
        try:
            cv2.moveWindow('Drone Camera Feed', 1200, 600)  # x=1200 (right), y=600 (lower, below LiDAR)
        except:
            pass  # Fallback if positioning fails

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.CTRL_FREQ>5 and i%10==0 and i/env.CTRL_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Capture and display camera feed #######################
        if show_camera and i % camera_update_freq == 0:
            try:
                # Get camera image from first drone
                rgb, depth, seg = env._getDroneImages(0, segmentation=False)
                
                # Convert RGBA to BGR for OpenCV
                rgb_bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
                
                # Resize to match window size (same width as LiDAR)
                display_size = (CAMERA_WINDOW_WIDTH, CAMERA_WINDOW_HEIGHT)
                rgb_display = cv2.resize(rgb_bgr, display_size)
                
                # Add text overlay with frame info
                cv2.putText(rgb_display, f'Frame: {i}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(rgb_display, f'Time: {i/env.CTRL_FREQ:.1f}s', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display in OpenCV window
                cv2.imshow('Drone Camera Feed', rgb_display)
                cv2.waitKey(1)  # Non-blocking, allows GUI to update
                
                # Ensure window stays in position (reposition if moved)
                try:
                    cv2.moveWindow('Drone Camera Feed', 1200, 600)
                except:
                    pass
                
            except Exception as e:
                # Camera might not be available yet or error occurred
                if i == 0:  # Only print once
                    print(f"[WARNING] Camera feed not available: {e}")

        #### Capture and visualize LiDAR scan #######################
        if show_lidar and i % lidar_update_freq == 0:
            try:
                # Get LiDAR scan from first drone
                ranges, hit_points, ray_angles = env._getDroneLidarScan(0)
                
                # Update polar plot visualization
                lidar_ax.clear()
                
                # Reapply orientation settings (clear() resets them)
                lidar_ax.set_theta_offset(np.pi/2)  # Rotate so 0° (forward) is at top
                lidar_ax.set_theta_direction(-1)  # Angles increase clockwise
                
                # Plot LiDAR data
                angles_rad = ray_angles[:, 0]  # Already in radians
                lidar_ax.plot(angles_rad, ranges, 'b.', markersize=2, label='LiDAR Scan')
                lidar_ax.set_ylim(0, env.LIDAR_MAX_RANGE)
                lidar_ax.set_title(f'2D LiDAR Scan - Frame {i}, Time {i/env.CTRL_FREQ:.1f}s', pad=20)
                lidar_ax.grid(True)
                
                # Update plot
                plt.draw()
                plt.pause(0.001)  # Small pause to allow plot to update
                
            except Exception as e:
                # LiDAR might not be available yet or error occurred
                if i == 0:  # Only print once
                    print(f"[WARNING] LiDAR scan not available: {e}")

        #### Compute control for the current way point #############
        for j in range(num_drones):
            if USE_CONTINUOUS_TRAJECTORY:
                # For continuous trajectories, compute position dynamically based on step count
                # This allows patterns to continue indefinitely without resetting
                # Use the waypoint pattern but adjust for continuous progression
                wp_idx = wp_counters[j] % NUM_WP
                cycles_completed = wp_counters[j] // NUM_WP
                
                # Get base position from waypoint pattern
                base_pos = TARGET_POS[wp_idx, :]
                # For Z coordinate, add the height gained per cycle times number of cycles
                z_per_cycle = TARGET_POS[-1, 2] - TARGET_POS[0, 2]  # Height change in one cycle
                continuous_z = INIT_XYZS[j, 2] + base_pos[2] + z_per_cycle * cycles_completed
                target_pos_3d = np.array([base_pos[0], base_pos[1], continuous_z])
                
                # Compute target yaw to face tangent to the 2D circular motion
                # For helix: the 2D projection is a circle, so yaw aligns with tangent to circle
                # Account for completed cycles to get continuous yaw
                t_base = (wp_idx / NUM_WP) * 4 * np.pi  # Angle within current cycle
                t_total = t_base + cycles_completed * 4 * np.pi  # Total angle including cycles
                target_yaw = t_total + np.pi/2  # Yaw = t + pi/2 for counterclockwise motion
                target_rpy_3d = np.array([INIT_RPYS[j, 0], INIT_RPYS[j, 1], target_yaw])
            else:
                # For closed-loop trajectories, use waypoint directly
                target_pos_3d = TARGET_POS[wp_counters[j], :] + np.array([0, 0, INIT_XYZS[j, 2]])
                # Use pre-computed target orientation if available, otherwise use initial
                if np.any(TARGET_RPY):  # Check if TARGET_RPY has been populated
                    target_rpy_3d = TARGET_RPY[wp_counters[j], :]
                else:
                    target_rpy_3d = INIT_RPYS[j, :]
            
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=target_pos_3d,
                                                                    # Alternative: Use only X,Y from waypoint, keep initial Z
                                                                    # target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                    target_rpy=target_rpy_3d
                                                                    )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            if USE_CONTINUOUS_TRAJECTORY:
                # For continuous trajectories, use modulo to repeat pattern seamlessly
                wp_counters[j] = (wp_counters[j] + 1) % NUM_WP
            else:
                # For closed-loop trajectories, wrap at end
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()
    
    #### Close OpenCV windows ##################################
    if show_camera:
        cv2.destroyAllWindows()
    if show_lidar:
        plt.close('all')  # Close all matplotlib figures

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--show_camera',        default=DEFAULT_SHOW_CAMERA, type=str2bool,      help='Whether to show live camera feed in OpenCV window (default: True)', metavar='')
    parser.add_argument('--camera_update_freq', default=DEFAULT_CAMERA_UPDATE_FREQ, type=int,  help='Update camera every N control steps (default: 5)', metavar='')
    parser.add_argument('--show_lidar',         default=DEFAULT_SHOW_LIDAR, type=str2bool,      help='Whether to show live LiDAR scan visualization (default: True)', metavar='')
    parser.add_argument('--lidar_update_freq', default=DEFAULT_LIDAR_UPDATE_FREQ, type=int,  help='Update LiDAR every N control steps (default: 5)', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
