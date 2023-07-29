#!/bin/sh

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n2

export PATH=$HOME/miniconda3/bin:$PATH
export CONDA_PREFIX=$HOME/miniconda3/envs/vits-lithuanian
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

. activate base
conda activate vits

python train.py -c files/configs/lt.json -m lithuanian -w 2