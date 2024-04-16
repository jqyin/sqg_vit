# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#### Mode 2: ViT coupled with EnSF for all steps
#### Mode 3: ViT coupled with EnSF for all steps, with jump noises added to the data

import os
import time

from datamodule import PLDataModule
from module_coupled_prediction import PLModule
from pytorch_lightning.cli import LightningCLI

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=PLModule,
        datamodule_class=PLDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},  # MGM: save_config_overwrite=True, auto_registry=True
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)

    ### set re-normalization
    cli.model.set_renormalization(mean_norm, std_norm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)

    cli.trainer.predict(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config['model']['pretrained_path'])
    cli.model.gather_predictions()

if __name__ == "__main__":
    main()
