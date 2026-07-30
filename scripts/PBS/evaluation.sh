#!/bin/bash

#PBS -l walltime=01:30:00
#PBS -q shortGPUQ
#PBS -l select=1:mem=20gb

#6530

echo "Job started on $(date)"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

if [ -d $HOME/PLANTORV ]; then
    cd $HOME/PLANTORV
elif [ -d $HOME/plantorv ]; then
    cd $HOME/plantorv
else
    echo "Directory $HOME/{PLANTORV,plantorv} does not exist. Exiting."
    exit 1
fi

if [ -d ./venv12 ]; then
    source ./venv12/bin/activate
elif [ -d ./venv ]; then
    source ./venv/bin/activate
else
    echo "Virtual environment $HOME/{PLANTORV,plantorv}/{venv12,venv} does not exist. Exiting."
    exit 1
fi

python3 evaluation/run_evaluation.py