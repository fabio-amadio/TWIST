
# bash train_teacher_task.sh 0927_twist_teacher_task cuda:0

cd legged_gym/legged_gym/scripts


exptid=$1
device=$2

task_name="g1_priv_mimic_task"
proj_name="g1_priv_mimic_task"

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                # --resume \
                # --debug
                # --resumeid xxx
