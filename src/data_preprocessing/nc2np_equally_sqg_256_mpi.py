# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import os

import click
import numpy as np
import xarray as xr

from climax.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR
from mpi4py import MPI

comm = MPI.COMM_WORLD
#rank = comm.Get_local_rank()
#size = comm.Get_local_size()
rank = int(os.getenv('SLURM_LOCALID', 0))
size = int(os.getenv('SLURM_NTASKS_PER_NODE', 1))


STEPS_PER_DAY = 80 

def nc2np(path, variables, days, save_dir, partition, num_shards_per_day):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
        
        # Distribute workload among ranks
        chunk_size = len(days) // size
        remainder = len(days) % size
        start_index = rank * chunk_size + days[0]
        end_index = (rank + 1) * chunk_size + (1 if rank < remainder else 0) + days[0]
        days = range(start_index, end_index)

    climatology = {}

    constant_fields = ["land_sea_mask", "orography", "lattitude"]
    var = 'sqg'

    for day in days:


        np_vars = {}

        ps =['/mnt/bb/junqi/splited_2hr_per_file/sqg/sqg_{}.nc'.format(day)]

        ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable
        code = NAME_TO_VAR[var]

        assert len(ds[code].shape) == 4
        all_levels = ds["z"][:].to_numpy()
        all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)

        for level in all_levels:
            ds_level = ds.sel(z=[level])
            level = int(level)

            np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()

            if partition == "train":  # compute mean and std of each var in each year
                var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                if var not in normalize_mean:
                    normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                    normalize_std[f"{var}_{level}"] = [var_std_yearly]
                else:
                    normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                    normalize_std[f"{var}_{level}"].append(var_std_yearly)

            clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0)
            if f"{var}_{level}" not in climatology:
                climatology[f"{var}_{level}"] = [clim_yearly]
            else:
                climatology[f"{var}_{level}"].append(clim_yearly)

        assert STEPS_PER_DAY % num_shards_per_day == 0
        num_hrs_per_shard = STEPS_PER_DAY // num_shards_per_day
        for shard_id in range(num_shards_per_day):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
            os.path.join(save_dir, partition, f"{day}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        for var in normalize_mean.keys():
            if var not in constant_fields:
                normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            if var not in constant_fields:
                mean, std = normalize_mean[var], normalize_std[var]
                variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
                std = np.sqrt(variance)
                mean = mean.mean(axis=0)
                normalize_mean[var] = mean
                normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )


@click.command()
@click.option("--root_dir", type=click.Path(exists=True))
@click.option("--save_dir", type=str)
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "sqg",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "toa_incident_solar_radiation",
        "total_precipitation",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ],
)
@click.option("--start_train_day", type=int, default=100)
@click.option("--start_val_day", type=int, default=200)
@click.option("--start_test_day", type=int, default=250)
@click.option("--end_day", type=int, default=300)
@click.option("--num_shards", type=int, default=1)
def main(
    root_dir,
    save_dir,
    variables,
    start_train_day,
    start_val_day,
    start_test_day,
    end_day,
    num_shards,
):
    assert start_val_day > start_train_day and start_test_day > start_val_day and end_day > start_test_day
    train_days = range(start_train_day, start_val_day)
    val_days = range(start_val_day, start_test_day)
    test_days = range(start_test_day, end_day)

    if rank == 0:
        print('-- train_years: {}'.format(train_days))
        print('-- val_years: {}'.format(val_days))
        print('-- test_years: {}'.format(test_days))
        # exit()
        os.makedirs(save_dir, exist_ok=True)

    nc2np(root_dir, variables, train_days, save_dir, "train", num_shards)
   
    if rank == 0: 
        nc2np(root_dir, variables, val_days, save_dir, "val", num_shards)
        nc2np(root_dir, variables, test_days, save_dir, "test", num_shards)

        # save lat and lon data
        ps = glob.glob(os.path.join(root_dir, variables[0], f"*{train_days[0]}*.nc"))
        x = xr.open_mfdataset(ps[0], parallel=True)
        lat = x["x"].to_numpy()
        lon = x["y"].to_numpy()
        np.save(os.path.join(save_dir, "lat.npy"), lat)
        np.save(os.path.join(save_dir, "lon.npy"), lon)

comm.Barrier()
MPI.Finalize()


if __name__ == "__main__":
    main()
