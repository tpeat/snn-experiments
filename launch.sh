#!/bin/bash
#SBATCH -JSNN-exp
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 8GB
#SBATCH -G A100:1
#SBATCH -t 01:00:00
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3/2022.05.0.1
module load gcc/12.3.0
module load mvapich2/2.3.7-1
module load cuda/12.1.1

echo "Launching Training"

 ~/.conda/envs/torch/bin/python exp.py