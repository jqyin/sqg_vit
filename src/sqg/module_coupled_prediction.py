# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from Enscore import EnSF

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from sqg.model.arch import SQG_ViT
from lr_scheduler import LinearWarmupCosineAnnealingLR
from metrics import (
    mse,
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)
from sqg.model.pos_embed import interpolate_pos_embed

class PLModule(LightningModule):
    """Lightning module for training with the SQG_ViT model.

    Args:
        net (SQG_ViT): SQG_ViT model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        net: SQG_ViT, 
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        observation_path: str = "",
        coupled_prediction_output_path: str = "",
        add_jump_noises: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)
    
        self.test_step_outputs = []
        self.test_outputs_dict = {}

        self.intermediate_preds = None
        self.rmse_auto_reg_list = []
        self.rmse_auto_reg_ensf_list = []
        self.r2_auto_reg_list = []
        self.r2_auto_reg_ensf_list = []

        ## prepare the dict for storing the prediction results
        # self.results_dict = {}
        self.pred_np_list = []
        self.pv_truth_list = []
        self.batch_idx_list = []

        self.rmse_no_ensf = []
        self.rmse_ensf = []

        self.bool_enable_ensf = True
        self.bool_enable_ensf_at_initialization_only = False
        self.bool_add_jump_noise = add_jump_noises
        self.bool_save_preds = True

        if self.bool_add_jump_noise:
            print('--- Enabling Jump Noise for auto-regressive predictions correction')
            self.rsjump = np.random.RandomState(24) # fixed seed for observations

        if self.bool_enable_ensf:
            print('--- Enabling EnSF for auto-regressive predictions correction')

        self.nanals = 20 # ensemble members
        self.obserrstdev = 1. # ob error standard deviation in K

        # nature run created using sqg_run.py.
        filename_climo = observation_path # file name for nature run to draw obs
        filename_truth = observation_path # use same file for nature run and climo to demonstrate EnSF

        # fix random seed for reproducibility.
        self.rsobs = np.random.RandomState(42) # fixed seed for observations
        self.rsics = np.random.RandomState() # varying seed for initial conditions

        # get model info
        self.nc_climo = Dataset(filename_climo)
        # parameter used to scale PV to temperature units in the Ensf steps.
        self.scalefact = self.nc_climo.f*self.nc_climo.theta0/self.nc_climo.g
        # initialize qg model instances for each ensemble member.
        x = self.nc_climo.variables['x'][:]
        y = self.nc_climo.variables['y'][:]
        x, y = np.meshgrid(x, y)
        nx = len(x); ny = len(y)
        dt = self.nc_climo.dt
        pv_climo = self.nc_climo.variables['pv']

        # nature run
        self.nc_truth = Dataset(filename_truth)
        self.pv_truth = self.nc_truth.variables['pv']

        print(self.nc_truth.variables['pv'].shape)
        print(self.nc_climo.variables['pv'].shape)

        # initialize ensemble members
        self.pvens = np.empty((self.nanals,2,ny,nx),np.float32)
        print('-- shape of pvens : ', self.pvens.shape) 

        # initial conditions
        #obs
        nobs = nx*ny
        self.pvob = np.empty((2,nobs),float)
        print('-- shape of pvob : ', self.pvob.shape) #
        for nanal in range(self.nanals):
            self.pvens[nanal] = pv_climo[0] + np.random.normal(0,1000,size=(2,ny,nx))

        self.ncount = 0
        self.ntstart = 0
        self.init_std_x_state = (self.pvens.reshape(self.nanals,2*nx*ny)).std(axis=0)
        print('# ntime, pverr_a') #RMSE

        self.nx = nx
        self.ny = ny
        self.nobs = nobs

        self.coupled_prediction_output_path = coupled_prediction_output_path

        ### Jump Noise
        if self.bool_add_jump_noise:
            percentage = 0.5
            noiselevel=np.array((1500,1200,900,600))
            noise_idx = np.array((1,1 - percentage*0.1, 1 - percentage*0.3, 1 - percentage*0.6, 1-percentage))
            jump_dist = self.rsjump.uniform(0,1,300)
            jump = np.where(jump_dist > 1-percentage)[0]

            # Initialize an empty dictionary
            self.jump_idx = {}
            for i in range(len(noiselevel)):
                value = noiselevel[i]
                keys = np.where((jump_dist <= noise_idx[i]) & (jump_dist > noise_idx[i+1]))[0].tolist()
                for key in keys:
                    self.jump_idx[key] = value

            if jump[0] == 0:    
                jump = np.delete(jump, 0)

            try:
                # Attempt to delete the key-value pair
                del self.jump_idx[0]
            except KeyError:
                # Handle the case where the key does not exist
                print("Key '{}' does not exist in the dictionary.".format(0))

            self.jump = jump

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        # interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_renormalization(self, mean_norm, std_norm):
        self.renormalization = transforms.Normalize(mean_norm, std_norm)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        loss_dict, _ = self.net.forward(x, y, lead_times, variables, out_variables, [mse], lat=self.lat)
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict["loss"]
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts, batch_preds = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=log_postfix,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts, batch_preds = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
        )
        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        # return loss_dict
        self.test_outputs_dict[batch_idx] = {'batch_idx': batch_idx, 'loss_dict': loss_dict, 'predictions': batch_preds , 'labels': y}
        # self.test_step_outputs.append({'batch_idx': batch_idx, 'loss_dict': loss_dict, 'predictions': batch_preds , 'labels': y})
        # return {'loss_dict': loss_dict, 'predictions': batch_preds , 'labels': y}
        return loss_dict
    
    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

    def predict_step(self, batch: Any, batch_idx: int):
        transforms = self.denormalization
        transforms_renormalization = self.renormalization

        if batch_idx == 0: ## t=0 
            x, y, lead_times, variables, out_variables = batch
            # # calculate mean and std of y variable
            if self.bool_enable_ensf:
                #### initial updates from the EnSF model
                indxob = np.sort(self.rsobs.choice(self.nx * self.ny, self.nobs, replace=False))

                if self.bool_add_jump_noise:
                    if (batch_idx == self.jump).any():
                        jumpnoise = self.rsjump.normal(0, self.jump_idx[batch_idx], size=(1, 2, self.ny, self.nx))
                        jumpnoise_reshape = jumpnoise.reshape(2, self.ny * self.nx)

                for k in range(2):
                    # surface temp obs
                    if self.bool_add_jump_noise:
                        if (batch_idx == self.jump).any():
                            self.pvob[k] = self.scalefact * (self.pv_truth[0, k, :, :].ravel()[indxob] + jumpnoise_reshape[k])
                        else:
                            self.pvob[k] = self.scalefact * self.pv_truth[0, k, :, :].ravel()[indxob]
                    else:
                        self.pvob[k] = self.scalefact * self.pv_truth[0, k, :, :].ravel()[indxob]
     
                    self.pvob[k] += self.rsobs.normal(scale=self.obserrstdev, size=self.nobs)

                # EnSF update
                EnSF_Update = EnSF(n_dim=self.nx * self.ny * 2, ensemble_size=self.nanals, eps_alpha=0.05, device='mps', \
                                obs_sigma=self.obserrstdev, euler_steps=1000, scalefact=self.scalefact,
                                init_std_x_state=self.init_std_x_state, ISarctan=False)
                # create 1d state vector.
                xens = self.pvens.reshape(self.nanals, 2, self.nx * self.ny)

                print('--- Start EnSF state update ---')
                xens = EnSF_Update.state_update_normalized(x_input=xens.reshape(self.nanals, 2 * self.nx * self.ny), \
                                                            state_target_input=self.pv_truth[0].reshape(2 * self.nx * self.ny), \
                                                            obs_input=self.pvob.reshape(2 * self.nx * self.ny))
                xens = xens.cpu().numpy()
                # back to 3d state vector
                self.pvens = xens.reshape((self.nanals, 2, self.ny, self.nx))

                pvensmean_a = self.pvens.mean(axis=0)
                if self.bool_add_jump_noise:
                    if (batch_idx == self.jump).any():
                        pverr_a = (self.scalefact * (pvensmean_a - self.pv_truth[batch_idx] - jumpnoise)) ** 2
                    else:
                        pverr_a = (self.scalefact * (pvensmean_a - self.pv_truth[batch_idx])) ** 2
                else:
                    pverr_a = (self.scalefact * (pvensmean_a - self.pv_truth[batch_idx])) ** 2

                print('SQG with EnSF, T: 0, ', np.sqrt(pverr_a.mean()))
                self.rmse_ensf.append(np.sqrt(pverr_a.mean()))

                ### save results for batch_idx == 0
                if self.bool_save_preds:
                    self.pred_np_list.append(self.pvens.mean(axis=0))
                    self.pv_truth_list.append(self.pv_truth[batch_idx])
                    self.batch_idx_list.append(batch_idx)

                ### run ML model predictions for the 20 ensemble members
                y = torch.tensor(self.pv_truth[1], dtype=torch.float32)  # self.pv_truth at t1
                y = y.unsqueeze(0) # add the sample dimension: [2,64,64] -> [1,2,64,64]
                y = transforms_renormalization(y)

                for nanal in range(self.nanals):
                    x_single_ens = torch.tensor(self.pvens[nanal], dtype=torch.float32)
                    ## add 1 dim at the first dimension
                    x_single_ens = x_single_ens.unsqueeze(0)
                    x_single_ens = transforms_renormalization(x_single_ens)
    
                    _, preds = self.net.forward(x_single_ens, y, lead_times, variables, out_variables, metric=None, lat=self.lat)
                    preds = transforms(preds)
                    y = transforms(y)
                
                    # plug back the prediction after convert back to numpy
                    self.pvens[nanal] = preds.cpu().detach().numpy()[0, :, :, :]
                    
            else: # ML prediction without EnSF

                x = torch.tensor(self.pv_truth[0], dtype=torch.float32)  # self.pv_truth at t0
                x = x.unsqueeze(0) # add the sample dimension: [2,64,64] -> [1,2,64,64]
                x = transforms_renormalization(x)
                y = torch.tensor(self.pv_truth[1], dtype=torch.float32)  # self.pv_truth at t1
                y = y.unsqueeze(0) # add the sample dimension: [2,64,64] -> [1,2,64,64]
                y = transforms_renormalization(y)

   
                _, preds = self.net.forward(x, y, lead_times, variables, out_variables, metric=None, lat=self.lat)
                ### transform back to the original scale
                preds = transforms(preds)
                y = transforms(y)

                ### first y torch.Tensor with shape [1, 2, 64, 64] is equavalent to the self.pv_truth[1] Numpy [2, 64, 64]
                pverr_a = (self.scalefact * (preds.cpu().detach().numpy()[0] - self.pv_truth[1])) ** 2
                print('SQG NO EnSF, T: 0, rmse: {}'.format(np.sqrt(pverr_a.mean())))
                self.rmse_no_ensf.append(np.sqrt(pverr_a.mean()))

                if self.bool_save_preds:
                    print(preds.cpu().detach().numpy()[0].shape) # [2, 64, 64]
                    print(self.pv_truth[1].shape)                # [2, 64, 64]

                    self.pred_np_list.append(preds.cpu().detach().numpy()[0])
                    self.pv_truth_list.append(self.pv_truth[1])
                    self.batch_idx_list.append(batch_idx)

                self.intermediate_preds = preds ### already denormalized
                # exit()

        elif batch_idx >0 and batch_idx <=301:
            x, y, lead_times, variables, out_variables = batch

            if self.bool_enable_ensf: # start running EnSF from t=1 step
                print('--- running the EnSF steps for batch_idx: {}'.format(batch_idx))
                indxob = np.sort(self.rsobs.choice(self.nx*self.ny,self.nobs,replace=False))

                if self.bool_add_jump_noise:
                    if (batch_idx == self.jump).any():
                        jumpnoise = self.rsjump.normal(0, self.jump_idx[batch_idx], size=(1, 2, self.ny, self.nx))
                        jumpnoise_reshape = jumpnoise.reshape(2, self.ny * self.nx)

                for k in range(2):
                    if self.bool_add_jump_noise:
                        if (batch_idx == self.jump).any():
                            self.pvob[k] = self.scalefact * (self.pv_truth[batch_idx, k, :, :].ravel()[indxob] + jumpnoise_reshape[k])
                        else:
                            self.pvob[k] = self.scalefact * self.pv_truth[batch_idx, k, :, :].ravel()[indxob]
                    else:
                        self.pvob[k] = self.scalefact * self.pv_truth[batch_idx, k, :, :].ravel()[indxob]      
                    self.pvob[k] += self.rsobs.normal(scale=self.obserrstdev, size=self.nobs)

                # EnSF update
                EnSF_Update = EnSF(n_dim=self.nx * self.ny * 2, ensemble_size=self.nanals, eps_alpha=0.05, device='mps', \
                                obs_sigma=self.obserrstdev, euler_steps=1000, scalefact=self.scalefact,
                                init_std_x_state=self.init_std_x_state, ISarctan=False)
                # create 1d state vector.
                xens = self.pvens.reshape(self.nanals, 2, self.nx * self.ny)

                print('--- Start EnSF state update ---')
                xens = EnSF_Update.state_update_normalized(x_input=xens.reshape(self.nanals, 2 * self.nx * self.ny), \
                                                            state_target_input=self.pv_truth[batch_idx].reshape(2 * self.nx * self.ny), \
                                                            obs_input=self.pvob.reshape(2 * self.nx * self.ny))
                xens = xens.cpu().numpy()
                # back to 3d state vector
                self.pvens = xens.reshape((self.nanals, 2, self.ny, self.nx))
                # print('--- shape of self.pvens: {}'.format(self.pvens.shape))

                # print out analysis error, spread and innov stats for background
                pvensmean_a = self.pvens.mean(axis=0)
                if self.bool_add_jump_noise:
                    if (batch_idx == self.jump).any():
                        pverr_a = (self.scalefact * (pvensmean_a - self.pv_truth[batch_idx] - jumpnoise)) ** 2
                    else:
                        pverr_a = (self.scalefact * (pvensmean_a - self.pv_truth[batch_idx])) ** 2
                else:
                    pverr_a = (self.scalefact * (pvensmean_a - self.pv_truth[batch_idx])) ** 2
                print('T: {}, rmse: {}'.format(batch_idx, np.sqrt(pverr_a.mean())))   
                self.rmse_ensf.append(np.sqrt(pverr_a.mean()))

                ### save results for batch_idx >0 and batch_idx <=300
                if self.bool_save_preds:
                    self.pred_np_list.append(self.pvens.mean(axis=0))
                    self.pv_truth_list.append(self.pv_truth[batch_idx])
                    self.batch_idx_list.append(batch_idx)

                y = torch.tensor(self.pv_truth[batch_idx], dtype=torch.float32)  # self.pv_truth at t_batch_idx+1
                y = y.unsqueeze(0) # add the sample dimension: [2,64,64] -> [1,2,64,64]
                y = transforms_renormalization(y)

                ### predictions on the 20 ensemble members
                for nanal in range(self.nanals):
                    # convert to torch.tensor
                    x_single_ens = torch.tensor(self.pvens[nanal], dtype=torch.float32)
                    x_single_ens = x_single_ens.unsqueeze(0)
                    x_single_ens = transforms_renormalization(x_single_ens)
                    _, preds = self.net.forward(x_single_ens, y, lead_times, variables, out_variables, metric=None, lat=self.lat)
                    preds = transforms(preds)
                    self.pvens[nanal] = preds.cpu().detach().numpy()[0, :, :, :]
                y = transforms(y)
            else:
                y = torch.tensor(self.pv_truth[batch_idx+1], dtype=torch.float32)  # self.pv_truth at t_batch_idx+1
                y = y.unsqueeze(0) # add the sample dimension: [2,64,64] -> [1,2,64,64]
                y = transforms_renormalization(y)

                self.intermediate_preds = transforms_renormalization(self.intermediate_preds)
                # print(self.intermediate_preds)
                ### normalize again
                _, preds = self.net.forward(self.intermediate_preds, y, lead_times, variables, out_variables, metric=None, lat=self.lat)
                ## calculate R2 and RMSE
                preds = transforms(preds)
                y = transforms(y)

                pverr_a = (self.scalefact * (preds.cpu().detach().numpy()[0] - self.pv_truth[batch_idx+1])) ** 2
                print('SQG NO EnSF, T: {}, rmse: {} '.format(batch_idx, np.sqrt(pverr_a.mean())))
                self.rmse_no_ensf.append(np.sqrt(pverr_a.mean()))

                if self.bool_save_preds:
                    self.pred_np_list.append(preds.cpu().detach().numpy()[0])
                    self.pv_truth_list.append(self.pv_truth[batch_idx+1])
                    self.batch_idx_list.append(batch_idx)

                self.intermediate_preds = preds

        self.batch_idx_list.append(batch_idx)

    def gather_predictions(self):
        # save the self.rmse_ensf and self.rmse_no_ensf to csv
        print('--- Running the gather_predictions()')
        if self.bool_enable_ensf:
            df = pd.DataFrame({'rmse_ensf': self.rmse_ensf})
            if self.bool_add_jump_noise:
                print('--- Saving the rmse_ensf_jump_noise to csv file')
                df.to_csv(self.coupled_prediction_output_path+'/rmse_ensf_jump_noise.csv', index=False)
                print('--- Saved the rmse_ensf_jump_noise CSV to: {}'.format(self.coupled_prediction_output_path+'/rmse_ensf_jump_noise.csv'))
            else:
                print('--- Saving the rmse_ensf to csv file')
                df.to_csv(self.coupled_prediction_output_path+'/rmse_ensf.csv', index=False)
                print('--- Saved the rmse_ensf CSV to: {}'.format(self.coupled_prediction_output_path+'/rmse_ensf.csv'))    
            ### save the predictions; self.pred_np_list, self.pv_truth_list, self.batch_idx_list
            stacked_preds_np = np.stack(self.pred_np_list, axis=0)
   
            if self.bool_add_jump_noise:
                np.save(self.coupled_prediction_output_path+'/stacked_preds_np_ensf_jump_noise.npy', stacked_preds_np)
                print('--- Saved the stacked_preds_np_ensf_jump_noise.npy to: {}'.format(self.coupled_prediction_output_path+'/stacked_preds_np_ensf_jump_noise.npy'))
            else:
                np.save(self.coupled_prediction_output_path+'/stacked_preds_np_ensf.npy', stacked_preds_np)
                print('--- Saved the stacked_preds_np_ensf.npy to: {}'.format(self.coupled_prediction_output_path+'/stacked_preds_np_ensf.npy'))

            ## stack the list of Numpy arrays along the first dimension
            stacked_pv_truth_np = np.stack(self.pv_truth_list, axis=0)
            stacked_pv_truth_np = stacked_pv_truth_np.data
            if self.bool_add_jump_noise:
                np.save(self.coupled_prediction_output_path+'/stacked_pv_truth_np_ensf_jump_noise.npy', stacked_pv_truth_np)
                print('--- Saved the stacked_pv_truth_np_ensf_jump_noise.npy to: {}'.format(self.coupled_prediction_output_path+'/stacked_pv_truth_np_ensf_jump_noise.npy'))
            else:
                np.save(self.coupled_prediction_output_path+'/stacked_pv_truth_np_ensf.npy', stacked_pv_truth_np)
                print('--- Saved the stacked_pv_truth_np_ensf.npy to: {}'.format(self.coupled_prediction_output_path+'/stacked_pv_truth_np_ensf.npy'))

        elif self.bool_enable_ensf_at_initialization_only:
            df = pd.DataFrame({'rmse_ensf': self.rmse_ensf})
            df.to_csv(self.coupled_prediction_output_path+'/rmse_ensf_initial_only.csv', index=False)
            print('-- Saved the rmse_ensf_initial_only CSV to: {}'.format(self.coupled_prediction_output_path+'/rmse_ensf_initial_only.csv'))
  
            ### stack all numpy arrays along the first dimension
            stacked_numpy_preds = np.stack(self.pred_np_list, axis=0)
            np.save(self.coupled_prediction_output_path+'/stacked_numpy_preds_ensf_initial_only.npy', stacked_numpy_preds)
            print('-- Saved the stacked_numpy_preds_ensf_initial_only.npy to: {}'.format(self.coupled_prediction_output_path+'/stacked_numpy_preds_ensf_initial_only.npy'))

            # save the pv_truth_list
            stacked_pv_truth_np = np.stack(self.pv_truth_list, axis=0)
            stacked_pv_truth_np = stacked_pv_truth_np.data
            np.save(self.coupled_prediction_output_path+'/stacked_pv_truth_np_ensf_initial_only.npy', stacked_pv_truth_np)
            print('-- Saved the stacked_pv_truth_np_ensf_initial_only.npy to: {}'.format(self.coupled_prediction_output_path+'/stacked_pv_truth_np_ensf_initial_only.npy'))

        else: ## no EnSF or EnSF initialization
            df = pd.DataFrame({'rmse_no_ensf': self.rmse_no_ensf})
            df.to_csv(self.coupled_prediction_output_path+'/rmse_no_ensf.csv', index=False)
            print('-- Saved the rmse_no_ensf CSV to: {}'.format(self.coupled_prediction_output_path+'/rmse_no_ensf.csv'))
            ### stack all numpy arrays along the first dimension
            stacked_numpy_preds = np.stack(self.pred_np_list, axis=0)
            np.save(self.coupled_prediction_output_path+'/stacked_numpy_preds_no_ensf.npy', stacked_numpy_preds)
            print('-- Saved the stacked_numpy_preds_no_ensf.npy to: {}'.format(self.coupled_prediction_output_path+'/stacked_numpy_preds_no_ensf.npy'))
            # save the pv_truth_list
            stacked_pv_truth_np = np.stack(self.pv_truth_list, axis=0)
            stacked_pv_truth_np = stacked_pv_truth_np.data
            np.save(self.coupled_prediction_output_path+'/stacked_pv_truth_np_no_ensf.npy', stacked_pv_truth_np)
            print('-- Saved the stacked_pv_truth_np_no_ensf.npy to: {}'.format(self.coupled_prediction_output_path+'/stacked_pv_truth_np_no_ensf.npy'))