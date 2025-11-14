#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=gpu
#SBATCH --mem=200G

module purge
module load Python/3.11.3-GCCcore-12.3.0

# Create and load virtual environment
python3 -m venv $HOME/venvs/DvhN_rt
source $HOME/venvs/DvhN_rt/bin/activate

# Install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Move downloaded models and tokenizers to the /scratch directory
export HF_HOME="/scratch/$USER/.cache/huggingface/hub"
# Disable xet downloader, use HTTP downloader
export HF_HUB_ENABLE_XET=0
export HF_HUB_DISABLE_XET=1

#scripts/preprocess_stage.sh
scripts/features_corr.sh
#scripts/pca.sh