import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import rotations # Requires rotations.py
import os
import time

class Hexapod:
    """
    Class to interface with the Hexapod MuJoCo simulation (using mujoco v3+ API)
    and implement a basic tripod gait controller.
    """

    def __init__(self, xml_path="hexapod.xml"):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MJCF file not found at: {xml_path}")

        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.simend = 30.0

        self.render_enabled = False
        self.window = None
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = None

        self.start_pos = None
        self.x_last = 0.0

        # --- Gait Parameters (TUNED DOWN for stability) ---
        # Start with smaller movements
        self.COXA_SWING_MAG = 0.2    # Reduced from 0.4
        self.FEMUR_LIFT_MAG = 0.3    # Reduced from 0.6
        self.TIBIA_EXTEND_MAG = 0.25 # Reduced from 0.5
        self.FEMUR_STANCE_MAG = -0.1 # Reduced from -0.2
        self.TIBIA_STANCE_MAG = 0.0
        self.CROUCH_FEMUR = -0.3
        self.CROUCH_TIBIA = 0.6
        # Give more time for each pose transition
        self.sim_steps_per_action = 150 # Increased from 50

        self.steps = self._define_gait_steps()
        self.current_step_index = 0
        self.steps_since_last_gait_update = self.sim_steps_per_action # Ensure first step is applied immediately in controller


    def _init_render(self):
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        self.window = glfw.create_window(1200, 900, "Hexapod Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        if self.context is None:
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
        self.render_enabled = True

        self.cam.azimuth = 90
        self.cam.elevation = -25
        self.cam.distance = 2.0
        self.cam.lookat = np.array([0.0, 0.0, 0.1])


    def _define_gait_steps(self):
        """ Defines the 4 phases of the tripod gait using current MAG parameters. """
        neutral = [
            0, self.CROUCH_FEMUR, self.CROUCH_TIBIA, 0, self.CROUCH_FEMUR, self.CROUCH_TIBIA,
            0, self.CROUCH_FEMUR, self.CROUCH_TIBIA, 0, self.CROUCH_FEMUR, self.CROUCH_TIBIA,
            0, self.CROUCH_FEMUR, self.CROUCH_TIBIA, 0, self.CROUCH_FEMUR, self.CROUCH_TIBIA,
        ]
        t1 = neutral.copy() # Phase 1: A=Stance(Back), B=Swing(Fwd/Up)
        t1[0], t1[6] = -self.COXA_SWING_MAG, -self.COXA_SWING_MAG
        t1[12] = +self.COXA_SWING_MAG
        t1[1], t1[7], t1[13] = self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG
        t1[2], t1[8], t1[14] = self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG
        t1[3] = +self.COXA_SWING_MAG
        t1[9], t1[15] = -self.COXA_SWING_MAG, -self.COXA_SWING_MAG
        t1[4], t1[10], t1[16] = +self.FEMUR_LIFT_MAG, +self.FEMUR_LIFT_MAG, +self.FEMUR_LIFT_MAG
        t1[5], t1[11], t1[17] = +self.TIBIA_EXTEND_MAG, +self.TIBIA_EXTEND_MAG, +self.TIBIA_EXTEND_MAG

        t2 = neutral.copy() # Phase 2: Transition/All Down (Pushing Back)
        t2[0], t2[6] = -self.COXA_SWING_MAG, -self.COXA_SWING_MAG
        t2[12] = +self.COXA_SWING_MAG
        t2[3] = -self.COXA_SWING_MAG
        t2[9], t2[15] = +self.COXA_SWING_MAG, +self.COXA_SWING_MAG
        t2[1], t2[4], t2[7], t2[10], t2[13], t2[16] = self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG
        t2[2], t2[5], t2[8], t2[11], t2[14], t2[17] = self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG

        t3 = neutral.copy() # Phase 3: B=Stance(Back), A=Swing(Fwd/Up)
        t3[3] = -self.COXA_SWING_MAG
        t3[9], t3[15] = +self.COXA_SWING_MAG, +self.COXA_SWING_MAG
        t3[4], t3[10], t3[16] = self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG, self.FEMUR_STANCE_MAG
        t3[5], t3[11], t3[17] = self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG, self.TIBIA_STANCE_MAG
        t3[0], t3[6] = +self.COXA_SWING_MAG, +self.COXA_SWING_MAG
        t3[12] = -self.COXA_SWING_MAG
        t3[1], t3[7], t3[13] = +self.FEMUR_LIFT_MAG, +self.FEMUR_LIFT_MAG, +self.FEMUR_LIFT_MAG
        t3[2], t3[8], t3[14] = +self.TIBIA_EXTEND_MAG, +self.TIBIA_EXTEND_MAG, +self.TIBIA_EXTEND_MAG

        t4 = t2.copy() # Phase 4: Transition/All Down (Pushing Back) - same as Phase 2

        return [t1, t2, t3, t4]

    # --- Core Logic Methods ---

    def get_position(self):
        if self.start_pos is None:
            return self.data.qpos[:3].copy()
        return self.data.qpos[:3] - self.start_pos

    def get_orientation(self):
        return self.data.qpos[3:7]

    def get_joint_angles(self):
        return self.data.qpos[7:7+18]

    def get_touch_data(self):
        return self.data.sensordata

    def reward(self):
        current_x = self.data.qpos[0]
        reward_val = current_x - self.x_last
        self.x_last = current_x
        return reward_val

    def done(self):
        torso_z = self.data.qpos[2]
        # More robust check: also consider orientation (e.g., if flipped over)
        quat = self.get_orientation()
        # Check if z-axis of torso points downwards significantly (using rotation matrix)
        # mat = np.zeros(9)
        # mj.mju_quat2Mat(mat, quat)
        # z_axis_global = mat[6:9] # Third column of rotation matrix
        # is_flipped = z_axis_global[2] < 0.5 # Example threshold for being flipped
        is_too_low = torso_z < 0.05
        return is_too_low # or (is_too_low or is_flipped)

    def observation(self):
        obs = list(self.get_position()) + \
              list(self.get_orientation_euler()) + \
              list(self.get_touch_data()) + \
              list(self.get_joint_angles())
        return np.array(obs)

    def _apply_action(self, action):
        """Internal method to apply normalized action to actuators."""
        if len(action) != 18:
             raise ValueError(f"Action length must be 18, got {len(action)}")
        if len(self.data.ctrl) != 18:
             raise RuntimeError(f"Expected 18 actuators, found {len(self.data.ctrl)}")

        target_angles_rad = np.clip(np.array(action), -1.0, 1.0) * (np.pi / 2.0)
        self.data.ctrl[:18] = target_angles_rad

    def get_orientation_euler(self):
        quat = self.get_orientation()
        try:
            return rotations.quat2euler(quat)
        except Exception as e:
            euler = np.zeros(3)
            mj.mju_quat2Euler(euler, quat)
            return euler

    def controller(self):
        """ Basic tripod gait controller, called every simulation step. """
        # Apply the *current* target pose repeatedly until it's time to switch
        action = self.steps[self.current_step_index]
        self._apply_action(action)

        # Check if it's time to update the gait phase *for the next step*
        self.steps_since_last_gait_update += 1
        if self.steps_since_last_gait_update >= self.sim_steps_per_action:
            self.current_step_index = (self.current_step_index + 1) % len(self.steps)
            self.steps_since_last_gait_update = 0
            # Debug print (optional)
            # print(f"Time: {self.data.time:.2f} Switched to Gait Step: {self.current_step_index}")


    def reset(self):
        """Resets the simulation and sets initial pose directly."""
        mj.mj_resetData(self.model, self.data)

        neutral_action_normalized = [
            0, self.CROUCH_FEMUR, self.CROUCH_TIBIA, 0, self.CROUCH_FEMUR, self.CROUCH_TIBIA,
            0, self.CROUCH_FEMUR, self.CROUCH_TIBIA, 0, self.CROUCH_FEMUR, self.CROUCH_TIBIA,
            0, self.CROUCH_FEMUR, self.CROUCH_TIBIA, 0, self.CROUCH_FEMUR, self.CROUCH_TIBIA,
        ]
        neutral_angles_rad = np.clip(np.array(neutral_action_normalized), -1.0, 1.0) * (np.pi / 2.0)

        if len(self.data.qpos) >= 7 + 18:
             self.data.qpos[7:7+18] = neutral_angles_rad
        else:
             print("Warning: Model qpos size doesn't match expected 7+18 DoFs.")

        # Set initial control to match initial pose
        self._apply_action(neutral_action_normalized)

        mj.mj_forward(self.model, self.data) # Compute initial state kinematics

        self.start_pos = self.data.qpos[:3].copy()
        self.x_last = self.start_pos[0]
        self.current_step_index = 0
        # Reset counter, controller will apply step 0 immediately
        self.steps_since_last_gait_update = self.sim_steps_per_action

        return self.observation()


    def simulate(self, render=True):
        """ Runs the simulation loop with integrated rendering. """
        if render and not self.render_enabled:
            self._init_render()
        elif not render and self.render_enabled:
            self.close()

        last_render_time = time.time()
        frame_rate = 60.0

        try:
            while not (self.render_enabled and glfw.window_should_close(self.window)):
                sim_start_time = self.data.time

                # Instability Check
                if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)) or np.max(np.abs(self.data.qacc)) > 1e6:
                     print(f"WARNING: Simulation unstable at Time = {self.data.time:.4f}. Stopping.")
                     # Print more info if needed
                     # print(f"Qpos: {self.data.qpos}")
                     # print(f"Qvel: {self.data.qvel}")
                     # print(f"Qacc: {self.data.qacc}")
                     break

                # Step simulation
                mj.mj_step(self.model, self.data)
                # Apply control based on current gait phase
                self.controller()

                if self.data.time >= self.simend:
                    print(f"Simulation ended at time {self.data.time:.2f}s")
                    break

                # Rendering
                if self.render_enabled:
                    current_time = time.time()
                    if current_time - last_render_time >= (1.0 / frame_rate):
                        viewport = mj.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
                        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL, self.scene)
                        mj.mjr_render(viewport, self.scene, self.context)
                        glfw.swap_buffers(self.window)
                        glfw.poll_events()
                        last_render_time = current_time

        except KeyboardInterrupt:
            print("Simulation interrupted.")
        except mj.FatalError as e:
             print(f"MuJoCo Fatal Error: {e}")
        finally:
            self.close()

    def close(self):
        """ Close the rendering window. """
        if self.render_enabled and self.window:
            glfw.destroy_window(self.window)
            self.window = None
            self.render_enabled = False
        # No need to free context explicitly in newer mujoco versions
        # if self.context:
        #    self.context = None # Just nullify the reference

    def __del__(self):
        self.close()