ROOT=/lustre/orion/world-shared/stf218/junqi/miniconda
export PATH=${ROOT}/bin:$PATH
source ${ROOT}/etc/profile.d/conda.sh
WORKDIR=$(pwd)
conda activate /lustre/orion/proj-shared/gen150/junqi/pyt2-env

export LD_LIBRARY_PATH=${ROOT}/lib:${ROOT}/../rccl-plugin-rocm540/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$WORKDIR/ClimaX/src:$PYTHONPATH
 


