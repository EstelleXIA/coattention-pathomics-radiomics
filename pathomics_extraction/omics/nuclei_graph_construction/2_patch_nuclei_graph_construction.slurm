#!/bin/bash
#SBATCH -J patch_graph
#SBATCH -p 64c512g
#SBATCH -o log/%a.out
#SBATCH -e log/%a.err
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH --array=0-100

module load miniconda3
source activate para-mil

echo $SLURM_ARRAY_TASK_ID
~/.conda/envs/para-mil/bin/python -u 2_patch_nuclei_graph_construction.py --task BLCA --idx $SLURM_ARRAY_TASK_ID
