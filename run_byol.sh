mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)
project_dir=$root_dir/projects/BYOL

export PYTHONPATH=$PYTHONPATH:${pwd}

task_id=byol_test

srun -u --partition=innova --job-name=${task_id} \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python ${project_dir}/main.py --gpus 1 2>&1 | tee log/${task_id}.log &
