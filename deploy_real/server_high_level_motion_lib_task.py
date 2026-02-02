#!/usr/bin/env python
import argparse
import time
import redis
import json
import numpy as np
import isaacgym
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive
# ---------------------------------------------------------------------
# Example imports: adapt to your actual file structure
# ---------------------------------------------------------------------
from isaacgym.torch_utils import quat_mul
from pose.utils.motion_lib_pkl import MotionLib
from pose.utils import torch_utils
from pose.util_funcs.kinematics_model import KinematicsModel
from legged_gym import LEGGED_GYM_ROOT_DIR
from data_utils.rot_utils import euler_from_quaternion, quat_rotate_inverse, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS, DEFAULT_ACTION_HAND

# ---------------------------------------------------------------------
# A small helper to replicate "mimic obs" logic from your code
# ---------------------------------------------------------------------
def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_obs_steps,
    task_body_names,
    kinematics_model=None,
    global_obs: bool = False,
):
    """
    Build the task-based mimic_obs at time-step t_step.
    """
    device = torch.device("cuda")
    # Build times
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_obs_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    # Suppose we only have a single motion in the .pkl
    motion_ids = torch.zeros(len(tar_obs_steps), dtype=torch.int, device=device)
    
    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, body_pos = motion_lib.calc_motion_frame(
        motion_ids, obs_motion_times
    )

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    if not global_obs:
        root_vel = quat_rotate_inverse_torch(root_rot, root_vel)
        root_ang_vel = quat_rotate_inverse_torch(root_rot, root_ang_vel)

    task_body_ids = motion_lib.get_key_body_idx(key_body_names=task_body_names)
    task_body_pos = body_pos[:, task_body_ids, :]
    if global_obs:
        root_rot_expand = root_rot.unsqueeze(1).expand(-1, task_body_pos.shape[1], -1)
        flat_root_rot = root_rot_expand.reshape(-1, 4)
        flat_task_pos = task_body_pos.reshape(-1, 3)
        flat_task_pos = torch_utils.quat_rotate(flat_root_rot, flat_task_pos)
        task_body_pos = flat_task_pos.reshape(task_body_pos.shape)
        task_body_pos = task_body_pos + root_pos.unsqueeze(1)

    task_body_rot = None
    local_body_rot = motion_lib.calc_local_body_rot(motion_ids, obs_motion_times)
    if local_body_rot is not None:
        task_body_rot = local_body_rot[:, task_body_ids, :]
        if global_obs:
            root_rot_expand = root_rot.unsqueeze(1).expand(-1, task_body_rot.shape[1], -1)
            flat_root_rot = root_rot_expand.reshape(-1, 4)
            flat_task_rot = task_body_rot.reshape(-1, 4)
            task_body_rot = quat_mul(flat_root_rot, flat_task_rot).reshape(task_body_rot.shape)
    else:
        if kinematics_model is None:
            raise RuntimeError(
                "local_body_rot missing in motion pkls and no kinematics model provided."
            )
        local_pos_fk, local_rot_fk, global_pos_fk, global_rot_fk = kinematics_model.forward_kinematics(
            dof_pos, root_pos, root_rot, task_body_names
        )
        task_body_rot = global_rot_fk if global_obs else local_rot_fk

    root_pos = root_pos.reshape(1, -1, 3)
    root_vel = root_vel.reshape(1, -1, 3)
    root_ang_vel = root_ang_vel.reshape(1, -1, 3)
    task_body_pos = task_body_pos.reshape(1, task_body_pos.shape[0], -1)
    task_body_rot = task_body_rot.reshape(-1, 4)
    task_body_rot = torch_utils.quat_to_tan_norm(task_body_rot)
    task_body_rot = task_body_rot.reshape(1, task_body_pos.shape[1], -1)

    mimic_obs_buf = torch.cat(
        (
            root_pos[..., 2:3],
            roll,
            pitch,
            yaw,
            root_vel[..., 0:2],
            root_ang_vel[..., 2:3],
            task_body_pos,
            task_body_rot,
        ),
        dim=-1,
    )[:, 0:1]
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    
    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


def main(args, xml_file, robot_base):

    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(sim_model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_JOINT, sim_model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(sim_model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            
    # 1. Connect to Redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # 2. Load motion library
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(args.motion_file, device=device)
    kinematics_model = None
    if not motion_lib.has_local_body_rot():
        kinematics_model = KinematicsModel(args.urdf, device=device)
    
    # 3. Prepare the steps array
    tar_obs_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_obs_steps_tensor = torch.tensor(tar_obs_steps, device=device, dtype=torch.int)
    task_body_names = [x.strip() for x in args.task_bodies.split(",") if x.strip()]

    # 4. Loop over time steps and publish mimic obs
    control_dt = 0.02
    # compute num_steps based on motion length
    motion_id = torch.tensor([0], device=device, dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)
    num_steps = int(motion_length / control_dt)
    
    print(f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds...")

    sample_mimic_obs, *_ = build_mimic_obs(
        motion_lib=motion_lib,
        t_step=0,
        control_dt=control_dt,
        tar_obs_steps=tar_obs_steps_tensor,
        task_body_names=task_body_names,
        kinematics_model=kinematics_model,
        global_obs=args.global_obs,
    )
    default_mimic_obs = DEFAULT_MIMIC_OBS.get(args.robot)
    if default_mimic_obs is None or default_mimic_obs.shape != sample_mimic_obs.shape:
        default_mimic_obs = np.zeros_like(sample_mimic_obs)

    last_mimic_obs = default_mimic_obs
    vis_root_vel = False
    vis_root_ang_vel = False
    if vis_root_vel:
        root_vel_list = []
    if vis_root_ang_vel:
        root_ang_vel_list = []
        
    try:
        for t_step in range(num_steps):
            t0 = time.time()

            # Build a mimic obs from the motion library
            mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
                motion_lib=motion_lib,
                t_step=t_step,
                control_dt=control_dt,
                tar_obs_steps=tar_obs_steps_tensor,
                task_body_names=task_body_names,
                kinematics_model=kinematics_model,
                global_obs=args.global_obs,
            )
            if vis_root_vel:
                root_vel_list.append(root_vel)
            if vis_root_ang_vel:
                root_ang_vel_list.append(root_ang_vel)

            # Convert to JSON (list) to put into Redis
            mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(mimic_obs_list))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            last_mimic_obs = mimic_obs
            # Print or log it
            print(f"Step {t_step:4d} => mimic_obs shape = {mimic_obs.shape} published...", end="\r")

            if args.vis:
                sim_data.qpos[:3] = root_pos
                # filp rot
                # root_rot = root_rot[[1,2,3,0]]
                root_rot = root_rot[[3,0,1,2]]
                sim_data.qpos[3:7] = root_rot
                sim_data.qpos[7:] = dof_pos
                mujoco.mj_forward(sim_model, sim_data)
                robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                # set distance to pelvis
                viewer.cam.distance = 2.0
                viewer.sync()
                
            # Sleep to maintain real-time pace
            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
        
    except KeyboardInterrupt:
        print("[Motion Server] Keyboard interrupt. Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (default_mimic_obs - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_mimic_{args.robot}", json.dumps(default_mimic_obs.tolist()))
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    finally:
        print("[Motion Server] Exiting...Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (default_mimic_obs - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_mimic_{args.robot}", json.dumps(default_mimic_obs.tolist()))
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", help="Path to your *.pkl motion file for MotionLib", 
                        default="/home/yanjieze/projects/g1_wbc/humanoid-motion-imitation/track_dataset/twist_motion_dataset/mocap/0.pkl")
    parser.add_argument("--robot", type=str, default="g1", choices=["g1"])
    parser.add_argument("--steps", type=str,
                        default="1",
                        help="Comma-separated steps for future frames (tar_obs_steps)")
    parser.add_argument("--task_bodies", type=str,
                        default="left_rubber_hand,right_rubber_hand",
                        help="Comma-separated task body names for mimic obs")
    parser.add_argument("--global_obs", action="store_true",
                        help="Use global frame for velocities and body poses")
    parser.add_argument(
        "--urdf",
        type=str,
        default=f"{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision_with_fixed_hand.urdf",
        help="URDF path for FK fallback if local_body_rot is missing",
    )
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    args = parser.parse_args()

    args.vis = True
    
    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)
    print("Task bodies: ", args.task_bodies)
    print("Global obs: ", args.global_obs)
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot == "g1":
        xml_file = f"{HERE}/../assets/g1/g1_mocap_with_wrist_roll.xml"
        robot_base = "pelvis"
    else:
        raise ValueError(f"robot type {args.robot} not supported")
    
    
    main(args, xml_file, robot_base)
