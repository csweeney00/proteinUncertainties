#!/bin/bash
# =====================
# Logging information
# =====================

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

source ~/.bashrc

# Make script bail out after first error
set -e

# Activate your conda environment
CONDA_ENV_NAME=enn-env
echo "Activating conda environment: ${CONDA_ENV_NAME}"
source /opt/conda/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
#eval "$(conda activate ${CONDA_ENV_NAME})"

# ==============================
# Finally, run the experiment!
# ==============================

experiment_text_file=$1
COMMAND="`sed '' ${experiment_text_file}`"
echo "Running ${COMMAND}"
eval "${COMMAND}"
echo "Command run successfully!"

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"