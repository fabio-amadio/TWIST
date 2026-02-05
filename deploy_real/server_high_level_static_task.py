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
# ---------------------------------------------------------------------
# Example imports: adapt to your actual file structure
# ---------------------------------------------------------------------
from legged_gym.envs.g1.task_obs_defs import TASK_MIMIC_OBS_DIM
from pose.util_funcs.kinematics_model import KinematicsModel
from pose.utils import torch_utils
from data_utils.rot_utils import euler_from_quaternion
from legged_gym import LEGGED_GYM_ROOT_DIR

def build_static_mimic_obs():
    device = torch.device("cpu")
    kinematics_model = KinematicsModel(
        f"{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision_with_fixed_hand.urdf",
        device=device,
    )

    # default pose (23 dof, no wrist roll)
    # dof_pos = torch.tensor(
    #     [
    #         -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg
    #         -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg
    #         0.0, 0.0, 0.0,  # torso
    #         0.0, 0.4, 0.0, 1.2,  # left arm
    #         0.0, -0.4, 0.0, 1.2,  # right arm
    #     ],
    #     dtype=torch.float32,
    #     device=device,
    # ).unsqueeze(0)
    dof_pos = torch.tensor(
        [
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg
            0.0, 0.0, 0.0,  # torso
            0.0, 0.0, 0.0, 0.0,  # left arm
            0.0, 0.0, 0.0, 0.0,  # right arm
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    root_pos = torch.tensor([0.0, 0.0, 0.793], dtype=torch.float32, device=device).unsqueeze(0)
    root_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device).unsqueeze(0)
    root_pos = root_pos.reshape(1, 1, 3)

    task_body_names = ["left_rubber_hand", "right_rubber_hand"]
    local_pos_fk, local_rot_fk, _, _ = kinematics_model.forward_kinematics(
        dof_pos, root_pos, root_rot, task_body_names
    )

    # roll, pitch, yaw = euler_from_quaternion(root_rot)
    # roll = roll.reshape(1, 1, 1)
    # pitch = pitch.reshape(1, 1, 1)
    # yaw = yaw.reshape(1, 1, 1)

    root_vel = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
    root_ang_vel = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
    task_body_pos = local_pos_fk.reshape(1, 1, -1)
    task_body_rot = local_rot_fk.reshape(-1, 4)
    task_body_rot = torch_utils.quat_to_tan_norm(task_body_rot).reshape(1, 1, -1)

    mimic_obs_buf = torch.cat(
        (
            root_pos[..., 2:3],  # 1 dim
            root_vel[..., 0:2],  # 2 dims, x, y only
            root_ang_vel[..., 2:3],  # 1 dim, yaw only
            task_body_pos,  # num_task_bodies * 3 dims
            task_body_rot,  # num_task_bodies * 6 dims
        ),
        dim=-1,
    )[:, 0:1]
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    if mimic_obs_buf.shape[1] != TASK_MIMIC_OBS_DIM:
        raise RuntimeError(
            f"Task mimic_obs dim mismatch: got {mimic_obs_buf.shape[1]}, "
            f"expected {TASK_MIMIC_OBS_DIM}"
        )
    return mimic_obs_buf.detach().cpu().numpy().squeeze().astype(np.float32)


def main(args, static_mimic_obs):

    # 1. Connect to Redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # 4. Loop over time steps and publish mimic obs
    control_dt = 0.02
    num_steps = int(args.run_time / control_dt)
    
    print(f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds...")

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
            mimic_obs = static_mimic_obs

            # Convert to JSON (list) to put into Redis
            mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
            redis_client.set(args.redis_mimic_key, json.dumps(mimic_obs_list))
            last_mimic_obs = mimic_obs
            # Print or log it
            print(f"Step {t_step:4d} => mimic_obs shape = {mimic_obs.shape} published...", end="\r")

                
            # Sleep to maintain real-time pace
            elapsed = time.time() - t0
            if t_step % 50 == 0:
                print(
                    f"[Timing] step={t_step} elapsed={elapsed*1000:.2f}ms "
                    f"budget={control_dt*1000:.2f}ms"
                )
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
        
    except KeyboardInterrupt:
        print("[Motion Server] Keyboard interrupt. Exiting without interpolation.")
        exit()
    except Exception:
        print("[Motion Server] Exception in streaming loop. Interpolating to default mimic_obs...")
        import traceback
        traceback.print_exc()
        # fall through to finally for interpolation + exit
    finally:
        print("[Motion Server] Exiting...")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="g1", choices=["g1"])
    parser.add_argument("--redis_mimic_key", type=str, default="action_mimic_task_g1",
                        help="Redis key for task mimic obs")
    parser.add_argument("--run_time", type=float, default=60.0, help="Run time in seconds")
    args = parser.parse_args()

    print("Robot type: ", args.robot)
    print("Run time: ", args.run_time)

    static_mimic_obs = build_static_mimic_obs()
    main(args, static_mimic_obs)
