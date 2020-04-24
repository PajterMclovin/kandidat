#!/bin/bash
# The name of the account you are running in, mandatory.
#SBATCH -A SNIC2020-5-120
# Request hh:mm:ss  of runtime for the job 
#SBATCH --time=24:00:00 
#SBATCH --error=./error_and_out/job.%J.err 
#SBATCH --output=./error_and_out/job.%J.out
# GPU node
#SBATCH--gres=gpu:k80:1

srun ./job_script.sh
