

#### Coupled prediction is done with local machine (Macbook pro) with 'mps' accelerator backend
#### NVIDIA/AMD GPU should work without any issue as well

export PYTHONPATH="${PYTHONPATH}:./src"
export PL_ACCELERATOR_BACKEND=mps

CMD="python ./src/sqg/vit_prediction.py --config ./configs/sqg_64_coupled.yaml \
    --trainer.strategy=single_device --trainer.devices=1 --trainer.num_nodes=1 --trainer.accelerator=mps \
    --trainer.max_epochs=1000 \
    --data.root_dir=./coupled_vit_ensf/sqg_N64_12hrly_npy \
    --data.predict_range=1 \
    --data.batch_size=1 \
    --model.pretrained_path=./coupled_vit_ensf/vit_model/last-v1.ckpt \
    --model.observation_path=./coupled_vit_ensf/sqg_N64_12hrly.nc \
    --model.coupled_prediction_output_path=./coupled_vit_ensf/coupled_results \
    --model.lr=1e-4 \
    --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5
"
echo $CMD

### run the command
$CMD

