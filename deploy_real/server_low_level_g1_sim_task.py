import argparse
import json
import time
import numpy as np
import redis
import mujoco
import isaacgym
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
from legged_gym.envs.g1.g1_mimic_distill_task_config import G1_MIMIC_OBS_DIM
from legged_gym.envs.g1.g1_specs import (
    G1_NUM_DOF,
    G1_DOF_NAMES,
    G1_DEFAULT_JOINT_ANGLES,
    G1_ANKLE_DOF_IDX,
    G1_XML_PATH,
)
from data_utils.rot_utils import quat_rotate_inverse

def draw_root_velocity(mujoco_model, mujoco_data, mujoco_viewer, tgt_root_vel, init_geom_id, root_name, rgba_velocity=[1, 1, 0, 1]):
    """
    Draws an arrow representing velocity, for debug/visualization.
    """
    mujoco_viewer.user_scn.ngeom = init_geom_id
    root_body_id = mujoco_model.body(root_name).id
    root_pos = mujoco_data.xpos[root_body_id]
    root_vel = tgt_root_vel
    vel_scale = 1.0

    mujoco.mjv_initGeom(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=rgba_velocity,
    )
    mujoco.mjv_connector(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=0.01,
        from_=root_pos,
        to=root_pos + vel_scale * np.array(root_vel),
    )
    mujoco_viewer.user_scn.ngeom += 1
    return mujoco_viewer.user_scn.ngeom


def quat_wxyz_rotate_inverse(quat_wxyz, vec):
    q_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float32,
    )
    return quat_rotate_inverse(q_xyzw[None, :], vec[None, :])[0]


def projected_gravity_from_quat(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


# -------------------------------------------------------------------
# Main low-level policy controller that:
#   - reads mimic obs from Redis
#   - feeds into policy
#   - runs the sim
# -------------------------------------------------------------------
    
class RealTimePolicyController:
    def __init__(self, 
                 xml_file, 
                 policy_path, 
                 device='cuda', 
                 record_video=False):
        
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device

        # Load policy
        self.policy = torch.jit.load(policy_path, map_location=device)
        print(f"Policy loaded from {policy_path}")

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            

        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=True, show_right_ui=True)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.cam.distance = 2.0

        # Example defaults & placeholders
        self.num_actions = G1_NUM_DOF
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        self.sim_decimation = 20

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # PD Gains, etc. (adapt as needed)
        self.default_dof_pos = np.array(
            [G1_DEFAULT_JOINT_ANGLES[name] for name in G1_DOF_NAMES],
            dtype=np.float32,
        )
        self.mujoco_default_dof_pos = np.concatenate(
            [
                np.array([0.0, 0.0, 0.793], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                self.default_dof_pos,
            ]
        )

        leg_kp = [100, 100, 100, 150, 40, 40] * 2
        waist_kp = [150, 150, 150]
        arm_kp = [40, 40, 40, 40]
        wrist_kp = [20, 20, 20]
        self.stiffness = np.array(
            leg_kp + waist_kp + arm_kp + wrist_kp + arm_kp + wrist_kp,
            dtype=np.float32,
        )

        leg_kd = [2, 2, 2, 4, 2, 2] * 2
        waist_kd = [4, 4, 4]
        arm_kd = [5, 5, 5, 5]
        wrist_kd = [1, 1, 1]
        self.damping = np.array(
            leg_kd + waist_kd + arm_kd + wrist_kd + arm_kd + wrist_kd,
            dtype=np.float32,
        )

        leg_tau = [88, 139, 88, 139, 50, 50] * 2
        waist_tau = [88, 50, 50]
        arm_tau = [25, 25, 25, 25]
        wrist_tau = [25, 25, 25]
        self.torque_limits = np.array(
            leg_tau + waist_tau + arm_tau + wrist_tau + arm_tau + wrist_tau,
            dtype=np.float32,
        )
        
        self.action_scale = 0.5
        self.clip_actions = 5.0
        self.action_clip = self.clip_actions / self.action_scale

        
        self.ankle_idx = list(G1_ANKLE_DOF_IDX)
        
        # For multi-step history
        self.n_mimic_obs = G1_MIMIC_OBS_DIM
        self.n_proprio = 3 + 3 + 3 + 3 * self.num_actions
        self.n_obs_single = self.n_mimic_obs + self.n_proprio
        self.proprio_history_buf = deque(maxlen=10)
        for _ in range(10):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single))

        self.record_video = record_video

    def extract_data(self):
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)

        dof_pos = qpos[7:7 + self.num_actions]
        dof_vel = qvel[6:6 + self.num_actions]

        quat = self.data.sensor('orientation').data.astype(np.float32)  # wxyz
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        lin_vel_world = qvel[0:3].astype(np.float32)
        return dof_pos, dof_vel, quat, ang_vel, lin_vel_world

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, mujoco_dof_pos=None):
        # body & hand
        self.data.qpos[:] = mujoco_dof_pos
        mujoco.mj_forward(self.model, self.data)
       
    def run(self):
        # Optionally record video
        if self.record_video:
            import imageio
            video_name = "debug_sim.mp4"
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating...")

        # send initial proprio to redis
        proprio_json = json.dumps(np.zeros(self.n_proprio, dtype=np.float32).tolist())
        self.redis_client.set("state_body_g1", proprio_json)
        self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))
        try:
            for i in pbar:
                
                t_start = time.time()
                dof_pos, dof_vel, quat, ang_vel, lin_vel_world = self.extract_data()
                
                if i % self.sim_decimation == 0:
                    
                    # Build a "proprio" vector for your policy, e.g.:
                    base_lin_vel = quat_wxyz_rotate_inverse(quat, lin_vel_world)
                    projected_gravity = projected_gravity_from_quat(quat)
                    obs_body_dof_vel = dof_vel.copy()
                    obs_body_dof_vel[self.ankle_idx] = 0.
                    obs_proprio = np.concatenate([
                        ang_vel * 0.25,
                        base_lin_vel,
                        projected_gravity,
                        (dof_pos - self.default_dof_pos),
                        obs_body_dof_vel * 0.05,
                        self.last_action
                    ])
                    # send proprio to redis
                    self.redis_client.set("state_body_g1", json.dumps(obs_proprio.tolist()))
                    self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))

                    # Try to get the latest mimic obs from Redis
                    try:
                        action_mimic_json = self.redis_client.get(self.redis_mimic_key)
                        if action_mimic_json is not None:
                            action_mimic_list = json.loads(action_mimic_json)
                            action_mimic = np.array(action_mimic_list, dtype=np.float32)
                        else:
                            raise Exception("cannot get action mimic from redis")
                    except:
                        raise Exception("cannot get action mimic from redis")

                    obs_full = np.concatenate([action_mimic, obs_proprio])
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    obs_buf = np.concatenate([obs_full, obs_hist])
                    self.proprio_history_buf.append(obs_full)

                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                    
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -self.action_clip, self.action_clip)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos
                    # debug draw velocity arrow if you want
                    self.viewer.user_scn.ngeom = 0
                    draw_root_velocity(self.model, self.data, self.viewer, [0,0,0], 0, "pelvis", [1,0,0,1])
                    
                    # make camera follow the pelvis
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()
                    if mp4_writer is not None:
                        img = self.viewer.read_pixels()
                        mp4_writer.append_data(img)

                # PD control
                torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                
                self.data.ctrl[:] = torque
                
                mujoco.mj_step(self.model, self.data)
                # sleep to maintain real-time pace
                elapsed = time.time() - t_start
                if elapsed < self.sim_dt:
                    time.sleep(self.sim_dt - elapsed)
        except Exception as e:
            print(f"Error in run: {e}")
            pass
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved")

            self.viewer.close()


def main_low_level_sim(args):
    controller = RealTimePolicyController(
        xml_file=args.xml_file,
        policy_path=args.policy_path,
        device='cuda',
        record_video=args.record_video,
    )
    controller.redis_mimic_key = args.redis_mimic_key
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_file", default=G1_XML_PATH, help="Mujoco XML file")
    
    parser.add_argument("--policy_path",  help="Path to the policy",
                        default="../assets/twist_general_motion_tracker.pt"
                        )
                        
    parser.add_argument("--record_video", action="store_true", help="Record a video")
    parser.add_argument("--redis_mimic_key", type=str, default="action_mimic_task_g1",
                        help="Redis key for task mimic obs")
    args = parser.parse_args()

    args.record_proprio = True
    
    main_low_level_sim(args)
