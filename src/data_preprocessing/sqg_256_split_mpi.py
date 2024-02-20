import netCDF4
import numpy as np
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
#rank = comm.Get_local_rank()
#size = comm.Get_local_size()
rank = int(os.getenv('SLURM_LOCALID', 0))
size = int(os.getenv('SLURM_NTASKS_PER_NODE', 1))

# Open the original netCDF file
filename = './data/sqg_N256_per_two_mininut.nc'  # Replace with your file name
ds = netCDF4.Dataset(filename, 'r')

# Time steps per day   90 seconds per time step
# steps_per_day = 960   # 1 file per day
steps_per_day = 80   # 1 file per 2 hours

# Calculate the total number of days
total_steps = len(ds.variables['t'])
total_days = (total_steps - 1) // steps_per_day

# Start day
start_day = 100

# Distribute workload among ranks
chunk_size = total_days // size
remainder = total_days % size
start_index = rank * chunk_size
end_index = (rank + 1) * chunk_size + (1 if rank < remainder else 0)

# Loop over each day and create a new file
for day in range(start_index, end_index):
    print(f'Rank {rank}: Processing day {day}')
    # Calculate the start and end index for slicing
    start_idx = day * steps_per_day
    end_idx = start_idx + steps_per_day

    # Create a new netCDF file for each day
    new_filename = f"/mnt/bb/junqi/splited_2hr_per_file/sqg/sqg_{start_day + day}.nc"
    with netCDF4.Dataset(new_filename, 'w', format='NETCDF4') as new_ds:
        # Copy dimensions
        for name, dimension in ds.dimensions.items():
            new_ds.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        # Copy variables
        for name, variable in ds.variables.items():
            new_var = new_ds.createVariable(name, variable.datatype, variable.dimensions)
            # Copy variable attributes
            new_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})

            # If it's the time variable, slice it
            if name == 't':
                new_var[:] = ds.variables['t'][start_idx:end_idx]
            # If it's the 'pv' variable (or any other multi-dimensional variable), slice it accordingly
            elif name == 'pv':
                new_var[:] = ds.variables['pv'][start_idx:end_idx, ...]
                print('-- start_idx: ', start_idx)
                print('-- end_idx: ', end_idx)
                print('-- total_steps: ', total_steps)
            else:
                new_var[:] = ds.variables[name][:]

print(f"Rank {rank}: Processing complete.")

# Close the original dataset
ds.close()

# Synchronize all ranks
comm.Barrier()

# If you want to see which rank is done last, uncomment below
# print(f"Rank {rank} is done.")

# Make sure all ranks finish before exiting
MPI.Finalize()

