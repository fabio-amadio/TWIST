import warnings

import torch

from isaacgym.torch_utils import quat_apply, quat_mul
import pytorch_kinematics as pk
def _quat_from_rot_matrix_batch(m: torch.Tensor) -> torch.Tensor:
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    c0 = trace > 0
    c1 = (~c0) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    c2 = (~c0) & (~c1) & (m[..., 1, 1] > m[..., 2, 2])
    c3 = ~(c0 | c1 | c2)

    S1 = torch.sqrt(trace + 1.0) * 2
    S2 = torch.sqrt(m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2] + 1.0) * 2
    S3 = torch.sqrt(m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2] + 1.0) * 2
    S4 = torch.sqrt(m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1] + 1.0) * 2

    w = torch.zeros_like(trace)
    x = torch.zeros_like(trace)
    y = torch.zeros_like(trace)
    z = torch.zeros_like(trace)

    w = torch.where(c0, S1 / 4, w)
    x = torch.where(c0, (m[..., 2, 1] - m[..., 1, 2]) / S1, x)
    y = torch.where(c0, (m[..., 0, 2] - m[..., 2, 0]) / S1, y)
    z = torch.where(c0, (m[..., 1, 0] - m[..., 0, 1]) / S1, z)

    x = torch.where(c1, S2 / 4, x)
    w = torch.where(c1, (m[..., 2, 1] - m[..., 1, 2]) / S2, w)
    y = torch.where(c1, (m[..., 0, 1] + m[..., 1, 0]) / S2, y)
    z = torch.where(c1, (m[..., 0, 2] + m[..., 2, 0]) / S2, z)

    y = torch.where(c2, S3 / 4, y)
    w = torch.where(c2, (m[..., 0, 2] - m[..., 2, 0]) / S3, w)
    x = torch.where(c2, (m[..., 0, 1] + m[..., 1, 0]) / S3, x)
    z = torch.where(c2, (m[..., 1, 2] + m[..., 2, 1]) / S3, z)

    z = torch.where(c3, S4 / 4, z)
    w = torch.where(c3, (m[..., 1, 0] - m[..., 0, 1]) / S4, w)
    x = torch.where(c3, (m[..., 0, 2] + m[..., 2, 0]) / S4, x)
    y = torch.where(c3, (m[..., 1, 2] + m[..., 2, 1]) / S4, y)

    q = torch.stack([x, y, z, w], dim=-1)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
    return q


class KinematicsModel:
    def __init__(self, file_path: str, device):
        self.device = device        
        if file_path.endswith(".urdf"):
            self.chain = pk.build_chain_from_urdf(open(file_path, mode="rb").read())
        elif file_path.endswith(".xml") or file_path.endswith(".mjcf"):
            self.chain = pk.build_chain_from_mjcf(open(file_path, mode="rb").read())
            
        self.chain = self.chain.to(device=device)
        self.num_joints = len(self.chain.get_joint_parameter_names())
        self.reindex = [12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        if len(self.reindex) != self.num_joints:
            self.reindex = list(range(self.num_joints))
        self._warned_reindex = False

        if len(set(self.reindex)) != len(self.reindex) or max(self.reindex, default=-1) >= self.num_joints:
            raise ValueError("Invalid joint reindex mapping for kinematics model.")
    
    def forward_kinematics(self, joint_angles: torch.Tensor, base_pos, base_rot, key_bodies: list):
        if not self._warned_reindex and self.reindex != list(range(self.num_joints)):
            warnings.warn(
                "KinematicsModel is using a hard-coded joint reindex without validation.",
                RuntimeWarning,
            )
            self._warned_reindex = True

        if joint_angles.device != self.device:
            joint_angles = joint_angles.to(self.device)
        if base_pos is not None and base_pos.device != self.device:
            base_pos = base_pos.to(self.device)
        if base_rot is not None and base_rot.device != self.device:
            base_rot = base_rot.to(self.device)

        if joint_angles.dtype != torch.float32:
            joint_angles = joint_angles.float()
        if base_pos is not None and base_pos.dtype != torch.float32:
            base_pos = base_pos.float()
        if base_rot is not None and base_rot.dtype != torch.float32:
            base_rot = base_rot.float()

        assert joint_angles.shape[1] == self.num_joints, (
            f"number of joints mismatch: {joint_angles.shape[1]} != {self.num_joints}"
        )
        joint_angles = joint_angles[:, self.reindex]

        local_pos = torch.zeros((joint_angles.shape[0], len(key_bodies), 3), device=self.device)
        local_rot = torch.zeros((joint_angles.shape[0], len(key_bodies), 4), device=self.device)

        ret = self.chain.forward_kinematics(joint_angles)

        for i, key_body in enumerate(key_bodies):
            tg = ret[key_body]
            m = tg.get_matrix()
            local_pos[:, i, :] = m[:, :3, 3]
            local_rot[:, i, :] = _quat_from_rot_matrix_batch(m[:, :3, :3])

        global_pos = None
        global_rot = None
        if base_pos is not None and base_rot is not None:
            flat_base_rot = base_rot.unsqueeze(1).expand(-1, local_rot.shape[1], -1).reshape(-1, 4)
            flat_local_rot = local_rot.reshape(-1, 4)
            global_rot = quat_mul(flat_base_rot, flat_local_rot).reshape(local_rot.shape)

            flat_local_pos = local_pos.reshape(-1, 3)
            flat_base_rot = (
                base_rot.unsqueeze(1)
                .expand(-1, local_pos.shape[1], -1)
                .reshape(-1, 4)
            )
            global_pos = quat_apply(flat_base_rot, flat_local_pos).reshape(local_pos.shape)
            global_pos = global_pos + base_pos.unsqueeze(1)

        return local_pos, local_rot, global_pos, global_rot
        
