import os
import pickle

import numpy as np


def _convert_file(path, out_path, mapping, joints_23, joints_29, idx_29, print_values=False):
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    dof_pos = data["dof_pos"]
    if not isinstance(dof_pos, np.ndarray):
        dof_pos = np.asarray(dof_pos)

    if dof_pos.shape[-1] != len(joints_23):
        raise ValueError(
            f"Expected dof_pos last dim {len(joints_23)}, got {dof_pos.shape[-1]} in {path}"
        )

    new_shape = dof_pos.shape[:-1] + (len(joints_29),)
    dof_pos_29 = np.zeros(new_shape, dtype=dof_pos.dtype)
    for i23, i29, _ in mapping:
        dof_pos_29[..., i29] = dof_pos[..., i23]

    if print_values:
        frame_idx = 0
        print(f"\nDOF values at frame {frame_idx} (23 vs 29):")
        for i23, i29, name in mapping:
            v23 = dof_pos[frame_idx, i23]
            v29 = dof_pos_29[frame_idx, i29]
            print(f"  {name:<26} 23[{i23:02d}]={v23:+.6f}  -> 29[{i29:02d}]={v29:+.6f}")

        print("\nExtra joints in 29 DOF (should be zero):")
        for name in joints_29:
            if name not in joints_23:
                i29 = idx_29[name]
                v29 = dof_pos_29[frame_idx, i29]
                print(f"  {name:<26} 29[{i29:02d}]={v29:+.6f}")

    data["dof_pos"] = dof_pos_29

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Saving: {out_path}")
    print(f"  dof_pos: {dof_pos.shape} -> {dof_pos_29.shape}")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)


def main():
    joints_23 = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    joints_29 = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    idx_29 = {name: i for i, name in enumerate(joints_29)}
    mapping = [(i, idx_29[name], name) for i, name in enumerate(joints_23)]

    print("23->29 DOF mapping:")
    for i23, i29, name in mapping:
        print(f"  [{i23:02d}] {name} -> [{i29:02d}]")
    folders = [
        "accad",
        "biomotionlab_ntroje",
        "bmlhandball",
        "bmlmovi",
        "cmu",
        "cnrs",
        "dancedb",
        "dfaust",
        "ekut",
        "eyes_japan",
        "grab",
        "hdm05",
        "human4d",
        "humaneva",
        "kit",
        "mocap",
        "mpi_limits",
        "mpi_mosh",
        "omomo",
        "sfu",
        "ssm_synce",
        "tcd_handmocap",
        "totalcapture",
        "transitions",
    ]

    first = True
    for folder in folders:
        in_dir = os.path.join("../track_dataset/twist_motion_dataset", folder)
        out_dir = os.path.join("../track_dataset/twist_motion_dataset_29dof", folder)
        if not os.path.isdir(in_dir):
            print(f"Skipping missing folder: {in_dir}")
            continue

        for name in sorted(os.listdir(in_dir)):
            if not name.endswith(".pkl"):
                continue
            in_path = os.path.join(in_dir, name)
            out_path = os.path.join(out_dir, name)
            _convert_file(
                in_path,
                out_path,
                mapping,
                joints_23,
                joints_29,
                idx_29,
                print_values=first,
            )
            first = False


if __name__ == "__main__":
    main()
