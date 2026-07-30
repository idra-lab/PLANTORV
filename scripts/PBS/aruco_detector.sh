#!/bin/bash

#PBS -l walltime=01:30:00
#PBS -q shortGPUQ
#PBS -l select=1:mem=20gb

#6530

echo "Job started on $(date)"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

source /home/i.delaossazarzuelo/venv12/bin/activate
cd /home/i.delaossazarzuelo/PLANTORV
python3 -u aruco/aruco_detector.py  --clean_dir dataset/rgb_aruco --tag_dir dataset/rgb_aruco --out_dir output_aruco --camera_yaml aruco/camera.yaml --config_yaml aruco/config.yaml 
