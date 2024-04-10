#!/bin/bash -l
#SBATCH -J sqg-vit
#SBATCH -N 4
#SBATCH -t 2:00:00
#SBATCH -A stf218
#SBATCH -C nvme
#SBATCH -q debug 
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -o log.o%j

source env.sh

DIST=deepspeed
SIZE=256
DATA=/mnt/bb/$USER/sqg_256_npy_2hr_per_file

export PL_DEEPSPEED_CONFIG_PATH=../configs/ds_config.yaml

CMD="PL_DEEPSPEED_CONFIG_PATH=../configs/ds_config.yaml \
    python ../src/sqg/train.py --config ../configs/sqg_${SIZE}.yaml  \
    --trainer.strategy=$DIST --trainer.devices=8 --trainer.num_nodes=$SLURM_NNODES \
    --trainer.max_epochs=10 \
    --data.root_dir=$DATA \
    --data.predict_range=1 \
    --data.batch_size=4 \
    --model.pretrained_path= \
    --model.lr=1e-6 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5
"

echo $CMD

HOME=/tmp time srun --nodes=${SLURM_NNODES} \
               --ntasks=$((SLURM_NNODES*8)) \
               --ntasks-per-node=8 \
               --gpu-bind=closest \
               -c7 \
               bash -c "source setup_dist.sh; $CMD"

