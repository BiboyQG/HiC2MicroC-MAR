#!/bin/bash
##############################################################################
#  SLURM DIRECTIVES
##############################################################################
#SBATCH --job-name=train-mar
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # one Python process
#SBATCH --cpus-per-task=64          # give that process all CPU cores
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --account=bcjw-delta-gpu
#SBATCH --mail-user=banghao2@illinois.edu
#SBATCH --mail-type="BEGIN,END"
#--- log files ---------------------------------------------------------------
#SBATCH --output=/projects/bewu/bchi/HiC2MicroC-MAR/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bewu/bchi/HiC2MicroC-MAR/slurm_logs/%x_%j.err
##############################################################################
#  JOB SCRIPT
##############################################################################

cd /projects/bewu/bchi/HiC2MicroC-MAR

uv run train-mar --config configs/mar_base_5kb.yaml
