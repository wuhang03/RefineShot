#!/bin/bash

# Exit on any error
set -e

# =================== Configuration Variables ===================
MODEL_NAME="ShotVL-3B"
NUM_GPUS=1
OUTPUT_DIR="eval_results"

OPENAI_API_KEY="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"  # Replace with your actual key or use environment variable
# ==============================================================

echo "Starting evaluation pipeline..."

# Derive prediction output path
PREDICTION_PATH="${OUTPUT_DIR}/${MODEL_NAME}"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

echo "Step 1: Running model evaluation with Accelerate"
accelerate launch --num_processes ${NUM_GPUS} evaluation/shotvl/evaluate.py --model ${MODEL_NAME} --reasoning --output-dir ${OUTPUT_DIR}

echo "Model evaluation completed. Predictions saved to: ${PREDICTION_PATH}"

echo "Step 2: Calculating scores using OpenAI API"
OPENAI_API_KEY=${OPENAI_API_KEY} python evaluation/calculate_scores.py --prediction_path ${PREDICTION_PATH}

echo "Score calculation completed."
echo "All steps finished successfully."