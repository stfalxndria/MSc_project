#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-power-smw
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0 

# Run calculations
timeout 1437m /work/e05/e05/stfalxndria/runCRYSTAL/Pcry_slurm diamond_64_2N2s 
/work/e05/e05/stfalxndria/runCRYSTAL/post_proc_slurm crys diamond_64_2N2s
