#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

module load cuda11.6/toolkit
python3 main.py --train_dqn
