
# bash eval_teacher_task.sh 0927_twist_teacher_task

task_name="g1_priv_mimic_task"
proj_name="g1_priv_mimic_task"
exptid=$1

cd legged_gym/legged_gym/scripts

# Run the eval script
python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --record_video \
