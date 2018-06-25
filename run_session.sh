#!/bin/bash

#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node 7
#SBATCH --gres=gpu:p100:1
#SBATCH -t 48:00:00

#echo commands to stdout
set -x

#move to working directory
# this job assumes:
# - all input data is stored in this directory 
# - all output should be stored in this directory 
cd "$SCRATCH/research/repos/im2txt_match"

#run GPU program
./eval.sh &
./init.sh
./fine_tune.sh


