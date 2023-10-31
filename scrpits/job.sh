#!/bin/bash -l
#SBATCH -J test
#SBATCH -N 4
#SBATCH -t 2:00:00
#SBATCH -A stf218
#SBATCH -C nvme
#SBATCH -q debug 
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -o tft.o%j

source env.sh

DIST=deepspeed

CMD="python ../src/climax/global_forecast/train.py --config configs/global_forecast_climax.yaml \
    --trainer.strategy=$DIST --trainer.devices=8 --trainer.num_nodes=$SLURM_NNODES \
    --trainer.max_epochs=10 \
    --data.root_dir=/lustre/orion/proj-shared/gen150/junqi/data/5.625deg_npz \
    --data.predict_range=72 --data.out_variables=['geopotential_500','temperature_850','2m_temperature'] \
    --data.batch_size=16 \
    --model.pretrained_path=../models/5.625deg.ckpt \
    --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5
"

echo $CMD

HOME=/tmp time srun --nodes=${SLURM_NNODES} \
               --ntasks=$((SLURM_NNODES*8)) \
               --ntasks-per-node=8 \
               --gpu-bind=closest \
               -c7 \
               bash -c "source setup_ddp.sh; $CMD"

