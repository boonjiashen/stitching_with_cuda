#!/bin/bash
#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00               # maximum run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=q2
#SBATCH -o %j.stdout

cd $SLURM_SUBMIT_DIR
cuda-memcheck ./cudaFeatureMatcher
#./testUnifiedMemory
