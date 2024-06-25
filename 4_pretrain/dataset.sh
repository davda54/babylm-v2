#!/bin/bash

#SBATCH --account=project_465000498
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=dev-g


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# Load the virtual environment
source /project/project_465000144/pytorch_1.13.1/bin/activate

# run the script
python3 dataset.py
