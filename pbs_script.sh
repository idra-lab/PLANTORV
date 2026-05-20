#!/bin/bash

#PBS -l walltime=00:30:00
#PBS -q shortGPUQ
#PBS -l select=1:mem=20gb:cpu_type=Xeon6140M

#6530

echo "Job started on $(date)"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

source /home/i.delaossazarzuelo/venv12/bin/activate
cd /home/i.delaossazarzuelo/describe-anything
python3 -u test_samgpt.py
# python3 clip.py