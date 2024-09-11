#!/bin/bash

# --account=stat-ecr
#SBATCH --qos=standard
#SBATCH --nodes=1
# --time=21-0:00:00
#SBATCH --job-name=TCR_pred
#SBATCH --mem=20000       

#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-opig-gpu
#SBATCH --nodelist=nagagpu02.cpu.stats.ox.ac.uk
#SBATCH --gres=gpu:1

# between swan01 for cpu and naga02 for gpu

# =====================
# Logging information
# =====================
echo "Job running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

#source ~/.bashrc

#set -e

# =============================
# Move the training directory
# =============================
USER_HOME=/data/localhost/not-backed-up/${USER}

echo "Moving data files"
#rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/" ${USER_HOME}/training_code/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/train_data_len_11.csv" ${USER_HOME}/training_code/MLAb/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/val_data_len_11.csv" ${USER_HOME}/training_code/MLAb/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/quick_train_data_len_11.csv" ${USER_HOME}/training_code/MLAb/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/quick_val_data_len_11.csv" ${USER_HOME}/training_code/MLAb/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/train.py" ${USER_HOME}/training_code/MLAb/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/data_storer.py" ${USER_HOME}/training_code/MLAb/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/util.py" ${USER_HOME}/training_code/MLAb/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/metrics.py" ${USER_HOME}/training_code/MLAb/
rsync -r "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/MLAb/config.csv" ${USER_HOME}/training_code/MLAb/

echo "Data files moved"

# ===================
# Environment setup
# ===================

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#CONDA_USER_HOME=/data/localhost/not-backed-up/chsweeney
#PATH_TO_CONDA="${CONDA_USER_HOME}/miniconda3"
#__conda_setup="$('${PATH_TO_CONDA}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "${PATH_TO_CONDA}/etc/profile.d/conda.sh" ]; then
#        . "${PATH_TO_CONDA}/etc/profile.d/conda.sh"
#    else
#        export PATH="${PATH_TO_CONDA}/bin:$PATH"
#    fi
#fi
#unset __conda_setup
# <<< conda initialize <<<

# Activate your conda environment
CONDA_ENV_NAME=env-abb2
bash
echo "Activating conda environment: ${CONDA_ENV_NAME}"
export PATH_TO_CONDA="${USER_HOME}/miniconda3"
source ${PATH_TO_CONDA}/bin/activate ${CONDA_ENV_NAME}
conda activate ${CONDA_ENV_NAME}


export WANDB_API_KEY=1c3160d56fb54c1bc4adb64e7f49f6699a798a45

conda info --envs


# =====================
# Run the experiment
# =====================

cd ${USER_HOME}/training_code

#rm ./models/train_data_ens_quick_len_11

#nvidia-smi

ens="3"
echo "ens ${ens}"

#python -u MLAb/train.py train_data_ens_${ens}_len_11 > train_data_ens_${ens}_len_11.txt
echo "Command run successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================

echo "Moving output data back to User Home"

rsync -r --archive --update --compress --progress "/data/localhost/not-backed-up/chsweeney/training_code/benchmarks/preds/ens_${ens}/" "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/benchmarks/preds/ens_${ens}/"
#rsync -r --archive --update --compress --progress "/data/localhost/not-backed-up/chsweeney/training_code/benchmarks/refined/ens_/" "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/benchmarks/refined/ens_3/"
rsync -r --archive --update --compress --progress "/data/localhost/not-backed-up/chsweeney/training_code/benchmarks/covars/ens_${ens}/" "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/benchmarks/covars/ens_${ens}/"
rsync -r --archive --update --compress --progress "/data/localhost/not-backed-up/chsweeney/training_code/train_data_ens_${ens}_len_11.txt" "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/"
rsync -r --archive --update --compress --progress "/data/localhost/not-backed-up/chsweeney/training_code/results_ens_${ens}_len_11.csv" "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/"
rsync -r --archive --update --compress --progress "/data/localhost/not-backed-up/chsweeney/training_code/models/" "/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/models/"


#'yes' | rm slurm*

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
