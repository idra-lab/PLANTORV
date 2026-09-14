#!/bin/sh

#SBATCH -p long
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --output=/home/enrico.saccon/plantorv/output_sbatch/output_%j.txt
#SBATCH --error=/home/enrico.saccon/plantorv/output_sbatch/error_%j.txt

date=$(date +%Y/%m/%d_%H:%M:%S)
echo "${date}"
echo "${date}" 1>&2;

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

cd /home/enrico.saccon/plantorv/
source /home/enrico.saccon/plantorv/.venv/bin/activate

./exec_eval.sh

# LLM_MODEL=hf_Qwen_Qwen3.6-35B-A3B && python3 evaluation/only_vlm.py --llm-config LLM/conf/${LLM_MODEL}.yaml --output-dir results_only_${LLM_MODEL} --aruco-dir output_aruco

