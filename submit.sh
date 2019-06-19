#!/bin/bash
#
#SBATCH -J "hyperband"
#SBATCH -p all
#SBATCH -t 5

python3 hyperband_demo.py --use_slurm=True
