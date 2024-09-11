#!/bin/bash

#SBATCH --array=10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150%5
#SBATCH --mem-per-cpu=3G
#SBATCH --clusters=srf_cpu_01
#SBATCH --partition=standard-cpu
#SBATCH --nodelist=naga01.cpu.stats.ox.ac.uk

#nodes = naga01, naga02, swan21
# naga02 doesn't have the env set up yet

# =====================
# Logging information
# =====================
echo "Job running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

source /homes/chsweeney/.bashrc

# Make script bail out after first error
#set -e

# Scratch Location
#SCRATCH_DISK=/data/localhost/not-backed-up/scratch
#SCRATCH_HOME=${SCRATCH_DISK}/${USER}
# User Directory Location
USER_HOME=/data/localhost/not-backed-up/${USER}
VOLS_HOME='/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents'

#echo "Moving conda env"
#rsync -r /vols/teaching/msc-projects/2023-2024/chsweeney/miniconda3 ${USER_HOME}
#echo "env moved"

echo "Moving data files"
#mkdir -p ${USER_HOME}/ToyExperiments
#mkdir -p ${USER_HOME}/models
rsync "${VOLS_HOME}/ToyExperiments/1d_regression.py" ${USER_HOME}/ToyExperiments/
rsync "${VOLS_HOME}/ToyExperiments/3d_regression.py" ${USER_HOME}/ToyExperiments/
rsync "${VOLS_HOME}/ToyExperiments/models.py" ${USER_HOME}/ToyExperiments/
rsync "${VOLS_HOME}/ToyExperiments/data_generation.py" ${USER_HOME}/ToyExperiments/
rsync "${VOLS_HOME}/ToyExperiments/plot_results.py" ${USER_HOME}/ToyExperiments/

#mkdir -p ${USER_HOME}/ToyExperiments/models/gap_10_models_3d
#rsync -r "${VOLS_HOME}/ToyExperiments/models/gap_10_models_3d/" ${USER_HOME}/ToyExperiments/models/gap_10_models_3d/

echo "Data files moved"

echo "Moving to node"
cd ${USER_HOME}/ToyExperiments

# Activate your conda environment
CONDA_ENV_NAME=enn-env
echo "Activating conda environment: ${CONDA_ENV_NAME}"
export PATH_TO_CONDA="${USER_HOME}/miniconda3"
#export PATH_TO_CONDA='/vols/teaching/msc-projects/2023-2024/chsweeney/miniconda3'
source ${PATH_TO_CONDA}/bin/activate ${CONDA_ENV_NAME}
conda activate ${CONDA_ENV_NAME}

export WANDB_API_KEY=1c3160d56fb54c1bc4adb64e7f49f6699a798a45

# =====================
# Run the experiment
# =====================

#python 3d_regression.py --lengthscale_idx ${SLURM_ARRAY_TASK_ID} --train_size 30 --data_size 20 --band_size 1 --inner_width 100 --lr 0.001 --num_epochs 300 --num_models 10 --batch_size 1 --ensemble_idx 0 --epoch_print_rate 10 --data_seed 30 --seed ${SLURM_ARRAY_TASK_ID} > toy_3d_experiment.txt
#python plot_results.py --train_size 30 --data_size 20 --three_d True --band_size 1 --batch_size 1 --inner_width 100 --data_seed 30 --seed 10 > toy_3d_experiment.txt

python 1d_regression.py --train_size ${SLURM_ARRAY_TASK_ID} --batch_size 20 --num_models 100 --ensemble_idx 0 --epoch_print_rate 100 --data_seed 25 --seed 5 --lr 0.01 --num_epochs 800 > toy_1d_experiment.txt
#python plot_results.py --batch_size 20 --data_seed 25 --seed 5 > toy_1d_experiment.txt

echo "Command run successfully!"

# ======================================
# Move output data from scratch to DFS
# ======================================

echo "Moving output data back to User Home"
rsync -r --archive --update --compress --progress '/data/localhost/not-backed-up/chsweeney/ToyExperiments/' '/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/ToyExperiments/'

# =============
# Cleaning Up
# =============

#echo "Cleaning the localhost data dir"

#rm -r ${USER_HOME}/training_code/data/*

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
