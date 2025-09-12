#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200
#SBATCH --time=2:00:00
#SBATCH --job-name=baseline
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.log


source /projects/sds-lab/Shuochen/miniconda3/bin/activate
conda activate ai_cuda118
python /home/wang.shuoc/downscaling/downscaling_github/baseline.py