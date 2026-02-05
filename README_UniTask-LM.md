
# UniTask‑LM (Unified Task‑Space Loco‑Manip)

UniTask‑LM unifies locomotion and manipulation by tracking task‑space references for the base and both hands. This makes the policy usable beyond teleop—e.g., as a low‑level controller for IK or generative planners.

## NOTES

- [ ] Training with the **G1-23dof** &rarr; hard to track EE orientation without wrist joints &rarr; **TODO**: use the **G1-29dof**
- [ ] Currently representing **rotations** using the **tangent and normal vectors** (not the first two columns of **R**)

## Installation

The training can be run on a single Nvidia RTX 4090 with 24G memory in 1~2 days.

**1**. Create **conda** environment:

```bash
conda env remove -n twist
conda create -n twist python=3.8
conda activate twist
```

**2**. Install **isaacgym**. Download from the [official link](https://developer.nvidia.com/isaac-gym) and then install it:

```bash
cd isaacgym/python && pip install -e .
```

If `import isaacgym` fails with `libpython3.8.so.1.0` missing, add the conda lib path on env activation:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh <<'EOF'
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#"$CONDA_PREFIX/lib:"}"
EOF
```

**3**. Install packages:

```bash
cd rsl_rl && pip install -e . && cd ..
cd legged_gym && pip install -e . && cd ..
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor
pip install redis[hiredis]
pip install pyttsx3 # for voice control
cd pose && pip install -e . && cd ..
```

Start redis server on your computer:

```bash
redis-server --daemonize yes
```

**4**. Download TWIST dataset from the provided link in the main README. Unzip it to anywhere you like, and set `root_path` in `legged_gym/motion_data_configs/twist_dataset.yaml` to the unzipped folder.

**5**. Ready for training & deployment.

## Usage

**1**. Training **task-based** teacher policy via RL:

```bash
bash train_teacher_task.sh 0101_twist_teacher_task cuda:0
```

**2**. Training **task-based** student policy via RL+BC (make sure  the teacher policy expid is the same above)

```bash
bash train_student_task.sh 0101_twist_rlbcstu_task 0101_twist_teacher_task cuda:0
```

**3**. Export student policy to jit model:

```bash
bash to_jit_task.sh 0101_twist_rlbcstu_task
```

You should see something like this:

```bash
Saved traced_actor at /PATH/TO/TWIST/legged_gym/logs/g1_stu_rl_task/0101_twist_rlbcstu_task/traced/0101_twist_rlbcstu-47500-jit.pt
Robot: g1
```

**4**. Sim2sim verification:
You can run the low-level simulation server.

```bash
cd deploy_real
python server_low_level_g1_sim_task.py --policy_path PATH/TO/YOUR/JIT/MODEL
```

- This will start a simulation that runs the low-level controller only.
- This is because we separate the high-level control (i.e., teleop) from the low-level control (i.e., RL policy).
- You should now be able to see the robot stand still.

And now you can control the robot via high-level motion server.

```bash
cd deploy_real
python server_high_level_motion_lib_task.py --motion_file /PATH/TO/MOTION.pkl
```

Use `--task_bodies` to override the default task bodies, `--global_obs` to switch to global frame, and `--urdf` if you need a custom URDF for FK fallback.

**5**. Sim2real verification:
Run the low-level controller first:

```bash
cd deploy_real
python server_low_level_g1_real_task.py --policy_path PATH/TO/YOUR/JIT/MODEL --net YOUR_NET_INTERFACE_TO_UNITREE_ROBOT
```

Then run the task-based high-level motion server:

```bash
cd deploy_real
python server_high_level_motion_lib_task.py --motion_file /PATH/TO/MOTION.pkl
```
