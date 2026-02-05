
# bash play_student_task.sh 0927_twist_rlbcstu_task 0927_twist_teacher_task

cd legged_gym/legged_gym/scripts

exptid=$1
teacher_exptid=$2
task_name="g1_stu_rl_task"
proj_name="g1_stu_rl_task"

# Run the eval script
python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --teacher_exptid "${teacher_exptid}" \
                --record_video \
