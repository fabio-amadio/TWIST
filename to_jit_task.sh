# bash to_jit_task.sh 0927_twist_rlbcstu_task

cd legged_gym/legged_gym/scripts


exptid=${1}

proj_name="g1_stu_rl_task"

# Run the training script
python save_jit_stu_rlbc_task.py \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --checkpoint -1 \
