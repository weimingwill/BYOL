mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)
project_dir=$root_dir/projects/byol

export PYTHONPATH=$PYTHONPATH:${pwd}

task_id=byol_test

srun -u --partition=innova --job-name=${task_id} \
    -n3 --gres=gpu:3 --ntasks-per-node=3 \
    python ${project_dir}/main.py --gpus 3 2>&1 | tee log/${task_id}.log &
