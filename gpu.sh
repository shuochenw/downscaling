#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --job-name=baseline
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.log

# Activate your conda environment
source /projects/sds-lab/Shuochen/miniconda3/bin/activate
conda activate ai_cuda118

# Run the code from the snapshot
# Change experiment: baseline.py, dann.py, full.py
python $SNAP_DIR/baseline.py
