from isaacgym.torch_utils import (
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
)

import torch

from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .g1_mimic_distill_task_config import G1MimicDistillTaskPrivCfg
from pose.utils import torch_utils
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import (
    convert_to_local_root_body_pos,
    convert_to_global_root_body_pos,
)
from pose.util_funcs.kinematics_model import KinematicsModel
from legged_gym.envs.g1.g1_specs import G1_ANKLE_DOF_IDX, G1_WAIST_DOF_IDX


class G1MimicDistillTask(HumanoidMimic):
    def __init__(
        self,
        cfg: G1MimicDistillTaskPrivCfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        self.cfg = cfg
        self.obs_type = cfg.env.obs_type
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        if self.obs_type == "student":
            self.total_env_steps_counter = 24 * 100000
            self.global_counter = 24 * 100000
            # self.motion_difficulty = torch.ones_like(self.motion_difficulty)

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(
                n, motion_difficulty=self.motion_difficulty
            )

        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(
                motion_ids.shape, device=self.device, dtype=torch.float
            )

        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            body_pos,
        ) = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        if body_pos.shape[1] != self._ref_body_pos.shape[1]:
            raise ValueError(
                "Motion/URDF body count mismatch at runtime: "
                f"motion={body_pos.shape[1]} vs urdf={self._ref_body_pos.shape[1]}."
            )
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(
            root_pos=root_pos, root_rot=root_rot, body_pos=body_pos
        )

    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            body_pos,
        ) = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        if body_pos.shape[1] != self._ref_body_pos.shape[1]:
            raise ValueError(
                "Motion/URDF body count mismatch at runtime: "
                f"motion={body_pos.shape[1]} vs urdf={self._ref_body_pos.shape[1]}."
            )
        self._ref_body_pos[:] = convert_to_global_root_body_pos(
            root_pos=root_pos, root_rot=root_rot, body_pos=body_pos
        )

    def _update_motion_difficulty(self, env_ids):
        if self.obs_type == "priv":
            super()._update_motion_difficulty(env_ids)
        elif self.obs_type == "student":
            super()._update_motion_difficulty(
                env_ids
            )  # currently we use the same strategy for student
        else:
            return

    def _init_buffers(self):
        super()._init_buffers()
        self.obs_history_buf = torch.zeros(
            (
                self.num_envs,
                self.cfg.env.history_len,
                self.cfg.env.n_obs_single,
            ),
            device=self.device,
        )
        self.privileged_obs_history_buf = torch.zeros(
            (
                self.num_envs,
                self.cfg.env.history_len,
                self.cfg.env.n_priv_obs_single,
            ),
            device=self.device,
        )
        self._task_body_names = list(self.cfg.motion.task_bodies)
        self._task_body_ids_motion = self._motion_lib.get_key_body_idx(
            key_body_names=self._task_body_names
        )
        self._task_body_ids = self._build_body_ids_tensor(self._task_body_names)
        self._kinematics_model = None
        if not self._motion_lib.has_local_body_rot():
            self._kinematics_model = KinematicsModel(
                self.cfg.asset.file, device=self.device
            )

    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(
            1, self.cfg.env.n_proprio, device=self.device
        )
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        ang_vel_dim = 3
        imu_dim = 2

        noise_scale_vec[:, 0:ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, ang_vel_dim : ang_vel_dim + imu_dim] = (
            self.cfg.noise.noise_scales.imu
        )
        noise_scale_vec[
            :, ang_vel_dim + imu_dim : ang_vel_dim + imu_dim + self.num_dof
        ] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[
            :,
            ang_vel_dim
            + imu_dim
            + self.num_dof : ang_vel_dim
            + imu_dim
            + 2 * self.num_dof,
        ] = self.cfg.noise.noise_scales.dof_vel

        return noise_scale_vec

    def _get_mimic_obs(self):
        num_steps = self._tar_obs_steps.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_obs_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(
            self._motion_ids.unsqueeze(-1), obs_motion_times.shape
        )
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            body_pos,
        ) = self._motion_lib.calc_motion_frame(
            motion_ids_tiled, obs_motion_times
        )

        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)

        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(
                root_pos=root_pos,
                root_rot=root_rot,
                body_pos=whole_key_body_pos,
            )
        whole_key_body_pos = whole_key_body_pos.reshape(
            self.num_envs, num_steps, -1
        )

        task_body_pos = body_pos[:, self._task_body_ids_motion, :]
        if self.global_obs:
            task_body_pos = convert_to_global_root_body_pos(
                root_pos=root_pos, root_rot=root_rot, body_pos=task_body_pos
            )
        task_body_pos = task_body_pos.reshape(self.num_envs, num_steps, -1)

        task_body_rot = None
        local_body_rot = self._motion_lib.calc_local_body_rot(
            motion_ids_tiled, obs_motion_times
        )
        if local_body_rot is not None:
            task_body_rot = local_body_rot[:, self._task_body_ids_motion, :]
            if self.global_obs:
                root_rot_expand = root_rot.unsqueeze(1).expand(
                    -1, task_body_rot.shape[1], -1
                )
                flat_root_rot = root_rot_expand.reshape(-1, 4)
                flat_task_rot = task_body_rot.reshape(-1, 4)
                task_body_rot = quat_mul(flat_root_rot, flat_task_rot).reshape(
                    task_body_rot.shape
                )
        else:
            if self._kinematics_model is None:
                raise RuntimeError(
                    "local_body_rot is missing from the motion pkl "
                    "and no kinematics model is available."
                )
            dof_pos = dof_pos.to(self.device)
            root_pos = root_pos.to(self.device)
            root_rot = root_rot.to(self.device)
            local_pos_fk, local_rot_fk, global_pos_fk, global_rot_fk = (
                self._kinematics_model.forward_kinematics(
                    dof_pos, root_pos, root_rot, self._task_body_names
                )
            )
            task_body_rot = global_rot_fk if self.global_obs else local_rot_fk

        root_pos = root_pos.reshape(
            self.num_envs, num_steps, root_pos.shape[-1]
        )
        root_vel = root_vel.reshape(
            self.num_envs, num_steps, root_vel.shape[-1]
        )
        root_ang_vel = root_ang_vel.reshape(
            self.num_envs, num_steps, root_ang_vel.shape[-1]
        )
        task_body_pos = task_body_pos.reshape(self.num_envs, num_steps, -1)
        task_body_rot = task_body_rot.reshape(-1, 4)
        task_body_rot = torch_utils.quat_to_tan_norm(task_body_rot)
        task_body_rot = task_body_rot.reshape(self.num_envs, num_steps, -1)
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])

        # teacher v0
        # shape: (num_envs, num_steps, 1+3+3+1+num_dof+3*num_key_bodies)
        priv_mimic_obs_buf = torch.cat(
            (
                root_pos[..., 2:3],  # 1 dim, z only
                roll,
                pitch,
                yaw,  # 3 dims
                root_vel,  # 3 dims
                root_ang_vel[..., 2:3],  # 1 dim, yaw only
                dof_pos,  # num_dof dims
                whole_key_body_pos,  # num_key_bodies * 3 dims
            ),
            dim=-1,
        )

        # shape: (num_envs, 1, 1+2+1+3*num_task_bodies+6*num_task_bodies)
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

        return priv_mimic_obs_buf.reshape(
            self.num_envs, -1
        ), mimic_obs_buf.reshape(self.num_envs, -1)

    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(
            0 * self.yaw, 0 * self.yaw, self.yaw
        )
        priv_mimic_obs, mimic_obs = self._get_mimic_obs()

        # shape: (num_envs, 3 + 2 + num_dof + num_dof + num_actions)
        proprio_obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3 dims
                imu_obs,  # 2 dims
                self.reindex(
                    (self.dof_pos - self.default_dof_pos_all)
                    * self.obs_scales.dof_pos
                ),  # num_dof dims
                self.reindex(
                    self.dof_vel * self.obs_scales.dof_vel
                ),  # num_dof dims
                self.reindex(self.action_history_buf[:, -1]),  # num_actions dims
            ),
            dim=-1,
        )

        if self.cfg.noise.add_noise and self.headless:
            proprio_obs_buf += (
                (2 * torch.rand_like(proprio_obs_buf) - 1)
                * self.noise_scale_vec
                * min(
                    self.total_env_steps_counter
                    / (self.cfg.noise.noise_increasing_steps * 24),
                    1.0,
                )
            )
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (
                2 * torch.rand_like(proprio_obs_buf) - 1
            ) * self.noise_scale_vec
        else:
            proprio_obs_buf += 0.0
        dof_vel_start_dim = 5 + self.dof_pos.shape[1]

        # disable ankle dof
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in G1_ANKLE_DOF_IDX]] = 0.0

        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(
                self.root_states[:, 3:7], key_body_pos
            )
        key_body_pos = key_body_pos.reshape(
            self.num_envs, -1
        )  # shape: (num_envs, num_key_bodies * 3)

        if self.cfg.domain_rand.domain_rand_general:
            priv_info = torch.cat(
                (
                    self.base_lin_vel,  # 3 dims
                    self.root_states[:, 2:3],  # 1 dim
                    key_body_pos,  # num_bodies * 3 dims
                    self.contact_forces[:, self.feet_indices, 2]
                    > 5.0,  # 2 dims, foot contact
                    self.mass_params_tensor,
                    self.friction_coeffs_tensor,
                    self.motor_strength[0] - 1,
                    self.motor_strength[1] - 1,
                ),
                dim=-1,
            )
        else:
            priv_info = torch.zeros(
                (self.num_envs, self.cfg.env.n_priv_info), device=self.device
            )

        obs_buf = torch.cat(
            (
                mimic_obs,
                proprio_obs_buf,
            ),
            dim=-1,
        )

        priv_obs_buf = torch.cat(
            (
                priv_mimic_obs,
                proprio_obs_buf,
                priv_info,
            ),
            dim=-1,
        )

        self.privileged_obs_buf = priv_obs_buf

        if self.obs_type == "priv":
            self.obs_buf = priv_obs_buf
        elif self.obs_type == "student":
            self.obs_buf = torch.cat(
                [obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1
            )

        if self.cfg.env.history_len > 0:
            self.privileged_obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([priv_obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat(
                    [
                        self.privileged_obs_history_buf[:, 1:],
                        priv_obs_buf.unsqueeze(1),
                    ],
                    dim=1,
                ),
            )
            if self.obs_type == "priv":
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            elif self.obs_type == "student":
                self.obs_history_buf = torch.where(
                    (self.episode_length_buf <= 1)[:, None, None],
                    torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                    torch.cat(
                        [self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)],
                        dim=1,
                    ),
                )

    ###########################################################################
    ###################### Extra Reward Functions##############################
    ###########################################################################

    def _reward_tracking_root_vel_xy(self):
        if self.global_obs:
            root_vel_diff = self._ref_root_vel[:, 0:2] - self.root_states[:, 7:9]
        else:
            local_ref_root_vel = quat_rotate_inverse(
                self._ref_root_rot, self._ref_root_vel
            )
            root_vel_diff = (
                local_ref_root_vel[:, 0:2] - self.base_lin_vel[:, 0:2]
            )
        root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
        return torch.exp(-1.0 * root_vel_err)

    def _reward_tracking_root_ang_vel_yaw(self):
        if self.global_obs:
            root_ang_vel_diff = (
                self._ref_root_ang_vel[:, 2] - self.root_states[:, 12]
            )
        else:
            local_ref_root_ang_vel = quat_rotate_inverse(
                self._ref_root_rot, self._ref_root_ang_vel
            )
            root_ang_vel_diff = (
                local_ref_root_ang_vel[:, 2] - self.base_ang_vel[:, 2]
            )
        root_ang_vel_err = root_ang_vel_diff * root_ang_vel_diff
        return torch.exp(-1.0 * root_ang_vel_err)

    def _reward_tracking_root_height(self):
        height_diff = self._ref_root_pos[:, 2] - self.root_states[:, 2]
        height_err = height_diff * height_diff
        return torch.exp(-5.0 * height_err)

    def _reward_tracking_task_body_pos(self):
        task_body_pos = (
            self.rigid_body_states[:, self._task_body_ids, 0:3]
            - self.root_states[:, 0:3].unsqueeze(1)
        )
        task_body_pos = convert_to_local_root_body_pos(
            self.root_states[:, 3:7], task_body_pos
        )
        tar_body_pos = (
            self._ref_body_pos[:, self._task_body_ids, :]
            - self._ref_root_pos.unsqueeze(1)
        )
        tar_body_pos = convert_to_local_root_body_pos(
            self._ref_root_rot, tar_body_pos
        )
        task_body_pos_diff = task_body_pos - tar_body_pos
        task_body_pos_err = torch.sum(task_body_pos_diff * task_body_pos_diff, dim=-1)
        task_body_pos_err = torch.sum(task_body_pos_err, dim=-1)
        return torch.exp(-10.0 * task_body_pos_err)

    def _reward_tracking_task_body_rot(self):
        task_body_rot = self.rigid_body_states[:, self._task_body_ids, 3:7]
        root_rot = self.root_states[:, 3:7]
        root_inv_rot = torch_utils.quat_conjugate(root_rot)
        root_inv_rot = root_inv_rot.unsqueeze(1).expand(-1, task_body_rot.shape[1], -1)
        flat_root_inv = root_inv_rot.reshape(-1, 4)
        flat_body_rot = task_body_rot.reshape(-1, 4)
        local_task_rot = quat_mul(flat_root_inv, flat_body_rot).reshape(
            task_body_rot.shape
        )

        motion_times = self._get_motion_times()
        ref_local_body_rot = self._motion_lib.calc_local_body_rot(
            self._motion_ids, motion_times
        )
        if ref_local_body_rot is None:
            if self._kinematics_model is None:
                return torch.zeros(self.num_envs, device=self.device)
            _, ref_local_body_rot, _, _ = self._kinematics_model.forward_kinematics(
                self._ref_dof_pos,
                self._ref_root_pos,
                self._ref_root_rot,
                self._task_body_names,
            )
        else:
            ref_local_body_rot = ref_local_body_rot[:, self._task_body_ids_motion, :]

        flat_ref_rot = ref_local_body_rot.reshape(-1, 4)
        flat_cur_rot = local_task_rot.reshape(-1, 4)
        rot_err = torch_utils.quat_diff_angle(flat_cur_rot, flat_ref_rot)
        rot_err = rot_err * rot_err
        rot_err = rot_err.reshape(self.num_envs, -1).sum(dim=1)
        return torch.exp(-5.0 * rot_err)

    def _reward_waist_dof_acc(self):
        waist_dof_idx = G1_WAIST_DOF_IDX
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[
                :, waist_dof_idx
            ],
            dim=1,
        )

    def _reward_waist_dof_vel(self):
        waist_dof_idx = G1_WAIST_DOF_IDX
        return torch.sum(torch.square(self.dof_vel[:, waist_dof_idx]), dim=1)

    def _reward_ankle_dof_acc(self):
        ankle_dof_idx = G1_ANKLE_DOF_IDX
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[
                :, ankle_dof_idx
            ],
            dim=1,
        )

    def _reward_ankle_dof_vel(self):
        ankle_dof_idx = G1_ANKLE_DOF_IDX
        return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)

    def _reward_ankle_action(self):
        return torch.norm(
            self.action_history_buf[:, -1, [4, 5, 10, 11]], dim=1
        )
