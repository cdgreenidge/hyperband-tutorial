#!/bin/bash
#
#SBATCH -J "hyperband"
#SBATCH -p all
#SBATCH -t 5

conda deactivate
conda activate
python3 hyperband_demo.py
