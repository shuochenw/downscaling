#!/bin/bash

# Snapshot directory (unique per submission, based on timestamp)
SNAP_DIR=/scratch/wang.shuoc/submission_snapshots/$(date +"%Y%m%d_%H%M%S")
mkdir -p $SNAP_DIR

# Copy the entire project (preserve folder structure)
cp -r /home/wang.shuoc/test/downscaling/* $SNAP_DIR/

# Submit the job and tell SLURM where the snapshot is
sbatch --export=SNAP_DIR=$SNAP_DIR gpu.sh