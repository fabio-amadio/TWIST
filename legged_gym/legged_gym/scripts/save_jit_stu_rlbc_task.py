import os, sys
import isaacgym

sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_mimic import Actor, get_activation
from legged_gym.envs.g1.g1_mimic_distill_task_config import G1_MIMIC_OBS_DIM
import argparse
from termcolor import cprint


def get_load_path(
    root, load_run=-1, checkpoint=-1, model_name_include="model"
):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [
            name
            for name in model_names
            if os.path.isdir(os.path.join(model_parent, name))
        ]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint == -1:
        models = [
            file for file in os.listdir(root) if model_name_include in file
        ]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model)
    return load_path, checkpoint


class HardwareRefNN(nn.Module):
    def __init__(
        self,
        num_obs,
        num_motion_observations,
        num_prop,
        num_actions,
        actor_hidden_dims=[512, 512, 256, 128],
        activation="silu",
        layer_norm=True,
        motion_latent_dim=128,
    ):
        super().__init__()

        self.num_prop = num_prop
        self.num_actions = num_actions
        self.num_obs = num_obs
        activation = get_activation(activation)

        self.normalizer = None

        self.actor = Actor(
            num_observations=num_obs,
            num_motion_observations=num_motion_observations,
            num_motion_steps=1,
            motion_latent_dim=motion_latent_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            tanh_encoder_output=False,
        )

    def load_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, obs):
        assert (
            obs.shape[1] == self.num_obs
        ), f"Expected {self.num_obs} but got {obs.shape[1]}"
        obs = self.normalizer.normalize(obs)
        return self.actor(obs)


def play(args):
    load_run = "../../logs/{}/{}".format(args.proj_name, args.exptid)
    checkpoint = args.checkpoint

    history_len = 10

    num_actions = 29
    n_proprio = 3 + 3 + 3 + 3 * num_actions
    n_mimic_obs = G1_MIMIC_OBS_DIM
    n_obs_single = n_mimic_obs + n_proprio
    num_observations = n_obs_single * (history_len + 1)

    device = torch.device("cpu")
    policy = HardwareRefNN(
        num_obs=num_observations,
        num_motion_observations=n_mimic_obs,
        num_prop=n_proprio,
        num_actions=num_actions,
    ).to(device)

    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    cprint(f"Loading model from: {load_path}", "green")
    ac_state_dict = torch.load(load_path, map_location=device)

    state_dict = ac_state_dict.get("model_state_dict", {})
    motion_w = state_dict.get("actor.motion_encoder.encoder.0.weight")
    actor0_w = state_dict.get("actor.actor_backbone.0.weight")
    motion_out_w = state_dict.get("actor.motion_encoder.linear_output.weight")
    if motion_w is not None and motion_w.shape[1] != n_mimic_obs:
        raise RuntimeError(
            f"Checkpoint n_mimic_obs={motion_w.shape[1]} does not match task setting "
            f"{n_mimic_obs}. Use a task-trained checkpoint."
        )
    if actor0_w is not None and motion_out_w is not None:
        inferred_num_obs = actor0_w.shape[1] - motion_out_w.shape[0]
        if inferred_num_obs != num_observations:
            raise RuntimeError(
                f"Checkpoint num_obs={inferred_num_obs} does not match task setting "
                f"{num_observations}. Check history_len or obs config."
            )
    policy.load_state_dict(ac_state_dict["model_state_dict"], strict=False)
    policy.load_normalizer(ac_state_dict["normalizer"])
    if (
        policy.normalizer is not None
        and policy.normalizer.get_shape()[0] != num_observations
    ):
        raise RuntimeError(
            f"Normalizer size {policy.normalizer.get_shape()[0]} does not match num_obs "
            f"{num_observations}. Use a task-trained checkpoint."
        )

    policy = policy.to(device)  # .cpu()
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    # Save the traced actor
    policy.eval()
    with torch.no_grad():
        num_envs = 2

        obs_input = torch.ones(num_envs, num_observations, device=device)
        print("obs_input shape: ", obs_input.shape)

        traced_policy = torch.jit.trace(policy, obs_input)

        # traced_policy = torch.jit.script(policy)
        save_path = os.path.join(
            load_run, "traced", args.exptid + "-" + str(checkpoint) + "-jit.pt"
        )
        traced_policy.save(save_path)
        cprint(f"Saved traced_actor at {os.path.abspath(save_path)}", "green")
        cprint("Robot: g1 (29-DOF)", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name", type=str)
    parser.add_argument("--exptid", type=str)
    parser.add_argument("--checkpoint", type=int, default=-1)

    args = parser.parse_args()
    play(args)
