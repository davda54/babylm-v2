#!/bin/bash
#SBATCH --job-name=babylm
#SBATCH --account=ec30
#SBATCH --time=24:00:00
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
# when running under SLURM control, i.e. as an actual batch job, box in NumPy
# (assuming we stick to the OpenBLAS back-end) to respect our actual allocation
# of cores.
#
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  export OPENBLAS_NUM_THREADS=${SLURM_CPUS_ON_NODE}
fi

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load Python/3.11.5-GCCcore-13.2.0

source pytorch/bin/activate

export PS1=\$

#export NCCL_SOCKET_IFNAME=eno1,eth0
export OMP_NUM_THREADS=6
export OPENBLAS_VERBOSE=2

export WANDB_MODE=online


trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM


echo $SLURM_GPUS_ON_NODE

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "



master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

#export NCCL_NSOCKS_PERTHREAD=4
#export NCCL_SOCKET_NTHREADS=2
#export NCCL_MIN_CHANNELS=32



# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
export NCCL_DEBUG=INFO


# by default, pass on any remaining command-line options
srun -W 0 python3 train_small.py 

PID="$!"
wait "${PID}"
