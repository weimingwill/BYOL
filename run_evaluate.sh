mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)
project_dir=$root_dir/projects/BYOL

export PYTHONPATH=$PYTHONPATH:${pwd}

task_id=evaluate_byol

srun -u --partition=innova --job-name=${task_id} \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python ${project_dir}/evaluate.py 2>&1 | tee log/${task_id}.log &
