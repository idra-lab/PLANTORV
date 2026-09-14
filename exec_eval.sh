#!/usr/bin/env bash

set -euo pipefail

SEG_MODELS=("sam3") # This must correspond to the names in the segmentation/conf folder
LLM_MODELS=("azure_gpt54-mini" "azure_gpt54-mini" "azure_gpt54-nano" "azure_claude-sonnet46") # This must correspond to the names in the LLM/conf folder
DEPTH_METHODS=("sensor" "monocular") # These are the --depth-source values, `python3 depth_estimation.py --help`.
DEPTH_ASSOCIATION_METHODS=("mask-median" "bbox-center") # These are the methods to associate the depth with the segmentation mask, `python3 depth_estimation.py --help`.

source .venv/bin/activate

# Without this, Python buffers stdout when it is piped, and the print() output of a
# run lands in its log out of order with the logger's stderr lines.
export PYTHONUNBUFFERED=1

# Run a command, showing its output on the terminal and saving it to a log file.
# Usage: run_logged <log-file> <command> [args...]
# tee writes the console copy through the script's own stdout rather than reopening
# /dev/stderr: a reopened stream has a file offset of its own, and when the script is
# redirected to a file (e.g. on PBS) it overwrites the echo lines, or is overwritten
# by them. The logger colours its level names with ANSI codes, which are stripped from
# the log once the command ends, whether or not it succeeded. A failing command still
# stops the script, through the status returned under `set -e`.
run_logged() {
    local log_file="$1"
    shift
    mkdir -p "$(dirname "${log_file}")"
    local status=0
    "$@" 2>&1 | tee "${log_file}" || status=$?
    sed -i 's/\x1b\[[0-9;]*m//g' "${log_file}"
    return "${status}"
}

# One line per timed step, printed together when the script exits. A trap rather than
# a final echo, so the summary also appears when a step fails and `set -e` stops the run.
TIMINGS=()
print_timings() {
    if [ "${#TIMINGS[@]}" -gt 0 ]; then
        echo "Timings:"
        printf '  %s\n' "${TIMINGS[@]}"
    fi
}
trap print_timings EXIT

# Like run_logged, and also measure the wall-clock time of the step, which includes
# interpreter start-up and model loading. The time goes to the terminal and to the end
# of the step's log, and is recorded even when the step fails.
# Usage: run_timed <step-name> <log-file> <command> [args...]
run_timed() {
    local step="$1"
    local log_file="$2"
    shift 2
    local start
    start=$(date +%s.%N)
    local status=0
    run_logged "${log_file}" "$@" || status=$?
    local elapsed
    elapsed=$(awk -v s="${start}" -v e="$(date +%s.%N)" \
        'BEGIN { t = e - s; printf "%.1fs (%dh %02dm %02ds)", t, t / 3600, (t % 3600) / 60, t % 60 }')
    local line="[TIME] ${step}: ${elapsed}"
    if [ "${status}" -ne 0 ]; then
        line="${line}, failed with exit code ${status}"
    fi
    echo "${line}" | tee -a "${log_file}"
    TIMINGS+=("${line}")
    return "${status}"
}


# First compute the ArUco markers for all images
run_logged output_aruco/aruco_detector.log \
    python3 aruco/aruco_detector.py \
        --clean_dir dataset/rgb_aruco \
        --tag_dir dataset/rgb_aruco \
        --out_dir output_aruco \
        --camera_yaml aruco/camera.yaml \
        --config_yaml aruco/config.yaml


# Then segment the RGB images
for SEG_MODEL in "${SEG_MODELS[@]}"; do
    echo "Running segmentation with model: ${SEG_MODEL}"
    SEG_DIR=results/output_segmentation_${SEG_MODEL}
    run_timed "segmentation ${SEG_MODEL}" ${SEG_DIR}/segmentation.log \
        python3 segmentation.py \
            --segmenter-config segmentation/conf/${SEG_MODEL}.yaml \
            --input-dir dataset/rgb \
            --output-dir ${SEG_DIR}
done


# Then compute the depth estimation for all images
for SEG_MODEL in "${SEG_MODELS[@]}"; do
    for LLM_MODEL in "${LLM_MODELS[@]}"; do
        # Run the annotation step with the segmentation model and LLM model
        echo "Running annotation with segmentation model: ${SEG_MODEL} and LLM model: ${LLM_MODEL}"
        ANNOTATION_DIR=results/results_${SEG_MODEL}_${LLM_MODEL}/annotations
        # The log sits inside annotations/, so the copy below carries it into every run directory.
        run_timed "annotation ${SEG_MODEL} ${LLM_MODEL}" ${ANNOTATION_DIR}/annotation.log \
            python3 annotation.py \
                --llm-config LLM/conf/${LLM_MODEL}.yaml \
                --input-dir results/output_segmentation_${SEG_MODEL}/output_segmentation \
                --output-dir ${ANNOTATION_DIR}

        for DEPTH_METHOD in "${DEPTH_METHODS[@]}"; do
            for DEPTH_ASSOCIATION_METHOD in "${DEPTH_ASSOCIATION_METHODS[@]}"; do
                RUN_DIR=results/results_${SEG_MODEL}_${LLM_MODEL}_${DEPTH_METHOD}_${DEPTH_ASSOCIATION_METHOD}
                mkdir -p ${RUN_DIR}/annotations
                cp -r ${ANNOTATION_DIR}/. ${RUN_DIR}/annotations/
            done
        done
        # # rm -rf results/results_${SEG_MODEL}_${LLM_MODEL}

        # Run depth estimation for each (depth source, association) pair
        for DEPTH_METHOD in "${DEPTH_METHODS[@]}"; do
            for DEPTH_ASSOCIATION_METHOD in "${DEPTH_ASSOCIATION_METHODS[@]}"; do
                echo "Running depth estimation with segmentation model: ${SEG_MODEL}, depth source: ${DEPTH_METHOD}, and depth association method: ${DEPTH_ASSOCIATION_METHOD}"
                RUN_DIR=results/results_${SEG_MODEL}_${LLM_MODEL}_${DEPTH_METHOD}_${DEPTH_ASSOCIATION_METHOD}
                run_timed "depth ${SEG_MODEL} ${LLM_MODEL} ${DEPTH_METHOD} ${DEPTH_ASSOCIATION_METHOD}" \
                    ${RUN_DIR}/annotations/depth_estimation.log \
                    python3 depth_estimation.py \
                        --input-dir results/output_segmentation_${SEG_MODEL}/output_segmentation \
                        --output-dir ${RUN_DIR}/annotations \
                        --depth-source ${DEPTH_METHOD} \
                        --depth-association ${DEPTH_ASSOCIATION_METHOD}
            done
        done
    done
done


# Finally run the evaluation for all the results
for SEG_MODEL in "${SEG_MODELS[@]}"; do
    for LLM_MODEL in "${LLM_MODELS[@]}"; do
        for DEPTH_METHOD in "${DEPTH_METHODS[@]}"; do
            for DEPTH_ASSOCIATION_METHOD in "${DEPTH_ASSOCIATION_METHODS[@]}"; do
                echo "Running evaluation with segmentation model: ${SEG_MODEL}, LLM model: ${LLM_MODEL}, depth source: ${DEPTH_METHOD}, and depth association method: ${DEPTH_ASSOCIATION_METHOD}"

                RUN_DIR=results/results_${SEG_MODEL}_${LLM_MODEL}_${DEPTH_METHOD}_${DEPTH_ASSOCIATION_METHOD}
                run_timed "evaluation ${SEG_MODEL} ${LLM_MODEL} ${DEPTH_METHOD} ${DEPTH_ASSOCIATION_METHOD}" \
                    ${RUN_DIR}/evaluation_results/evaluation.log \
                    python3 -m evaluation.run_evaluation \
                        --seg-dir ${RUN_DIR}/annotations/ \
                        --aruco-dir output_aruco \
                        --rgb-dir dataset/rgb \
                        --images 1-151 \
                        --output-dir ${RUN_DIR}/evaluation_results
            done
        done
    done
done
