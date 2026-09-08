#!/usr/bin/env bash

set -euo pipefail

SEG_MODELS=("sam21-l" "sam1-h" "fastsam-s" "mobile-sam" "sam3") # This must correspond to the names in the segmentation/conf folder
LLM_MODELS=("azure_gpt54-mini" "azure_gpt54-nano") # This must correspond to the names in the LLM/conf folder
DEPTH_METHODS=("sensor" "monocular") # These are the --depth-source values, `python3 depth_estimation.py --help`.
DEPTH_ASSOCIATION_METHODS=("bbox-center" "mask-median") # These are the methods to associate the depth with the segmentation mask, `python3 depth_estimation.py --help`.

source .venv/bin/activate


# # First compute the ArUco markers for all images
# python3 aruco/aruco_detector.py \
#     --clean_dir dataset/rgb_aruco \
#     --tag_dir dataset/rgb_aruco \
#     --out_dir output_aruco \  
#     --camera_yaml aruco/camera.yaml \
#     --config_yaml aruco/config.yaml


# # Then segment the RGB images
# for SEG_MODEL in "${SEG_MODELS[@]}"; do
#     echo "Running segmentation with model: ${SEG_MODEL}"
#     python3 segmentation.py \
#         --segmenter-config segmentation/conf/${SEG_MODEL}.yaml \
#         --input-dir dataset/rgb \
#         --output-dir results/output_segmentation_${SEG_MODEL}
# done


# Then compute the depth estimation for all images
for SEG_MODEL in "${SEG_MODELS[@]}"; do
    for LLM_MODEL in "${LLM_MODELS[@]}"; do
        # Run the annotation step with the segmentation model and LLM model
        echo "Running annotation with segmentation model: ${SEG_MODEL} and LLM model: ${LLM_MODEL}"
        python3 annotation.py \
            --llm-config LLM/conf/${LLM_MODEL}.yaml \
            --input-dir results/output_segmentation_${SEG_MODEL}/output_segmentation \
            --output-dir results/results_${SEG_MODEL}_${LLM_MODEL}/annotations

        for DEPTH_METHOD in "${DEPTH_METHODS[@]}"; do
            for DEPTH_ASSOCIATION_METHOD in "${DEPTH_ASSOCIATION_METHODS[@]}"; do
                RUN_DIR=results/results_${SEG_MODEL}_${LLM_MODEL}_${DEPTH_METHOD}_${DEPTH_ASSOCIATION_METHOD}
                mkdir -p ${RUN_DIR}/annotations
                cp -r results/results_${SEG_MODEL}_${LLM_MODEL}/annotations/. ${RUN_DIR}/annotations/
            done
        done
        # # rm -rf results/results_${SEG_MODEL}_${LLM_MODEL}

        # Run depth estimation for each (depth source, association) pair
        for DEPTH_METHOD in "${DEPTH_METHODS[@]}"; do
            for DEPTH_ASSOCIATION_METHOD in "${DEPTH_ASSOCIATION_METHODS[@]}"; do
                echo "Running depth estimation with segmentation model: ${SEG_MODEL}, depth source: ${DEPTH_METHOD}, and depth association method: ${DEPTH_ASSOCIATION_METHOD}"
                python3 depth_estimation.py \
                    --input-dir results/output_segmentation_${SEG_MODEL}/output_segmentation \
                    --output-dir results/results_${SEG_MODEL}_${LLM_MODEL}_${DEPTH_METHOD}_${DEPTH_ASSOCIATION_METHOD}/annotations \
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
