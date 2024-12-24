#!/bin/bash

#SBATCH --job-name=llama-3.2-1B-sft         # Job name
#SBATCH --output="job-%x-$(date)-%j.log"    # Name of stdout output file (%x expands to job name and %j expands to %jobId)
#SBATCH --time=48:00:00                     # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1                        # Number of GPUs to be used
#SBATCH --partition a6000                   # Partition to be used
#SBATCH --qos=gpu-medium                    # Quality of Service (QOS) to be used

eval "$(conda shell.bash hook)"
conda activate tunalm
conda env list
echo "${PWD}"

tune run tunalm/sft.py --config tunalm/configs/1B_sft.yaml
