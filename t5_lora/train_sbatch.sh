#!/bin/bash
#SBATCH -A research 
#SBATCH -c 38
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 2G
#SBATCH --time 2-00:00:00
#SBATCH --output fincausal_t5-small_loadin8bit.logs
#SBATCH --mail-user pavan.baswani@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name LegalFinetune

module load u18/cuda/10.2

source /home/sumukh.s/anaconda2/etc/profile.d/conda.sh

conda deactivate

echo "conda activate instructner"
conda activate instructner
echo "instructner environment activated"

echo "available cuda devices: "
echo "$CUDA_VISIBLE_DEVICES"

echo "exporting cuda device order"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo "exporting cuda devices numbers"
export CUDA_VISIBLE_DEVICES="0"

echo "training"
python3 run_ner.py
echo "training completed"

echo "deactivate instructner"
conda deactivate
echo "notebook finished"