#!/bin/bash
#SBATCH -J histomics_feature
#SBATCH -p 64c512g
#SBATCH -o log/%a.out
#SBATCH -e log/%a.err
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --ntasks-per-node=64
#SBATCH --array=0-100

module load miniconda3
source activate para-mil

echo $SLURM_ARRAY_TASK_ID
~/.conda/envs/para-mil/bin/python -u 1_nuclei_feature_extraction.py --task BLCA --idx $SLURM_ARRAY_TASK_ID
