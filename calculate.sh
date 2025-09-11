MODEL_NAME="InternVL3-8B"
MODEL_NAME="Qwen2.5-VL-7B"
# MODEL_NAME="ShotVL-7B"
NUM_GPUS=1
OUTPUT_DIR="eval_results"

CATEGORY="all"

FILENAME="predictions_1757518602.xlsx"

# PREDICTION_PATH="${OUTPUT_DIR}/${MODEL_NAME}/${FILENAME}"

PREDICTION_PATH="${OUTPUT_DIR}/${MODEL_NAME}/${FILENAME}"

OPENAI_API_KEY="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" 

echo "Step 2: Calculating scores using OpenAI API"
OPENAI_API_KEY=${OPENAI_API_KEY} python evaluation/calculate_scores.py --prediction_path ${PREDICTION_PATH} --category "${CATEGORY}" 

echo "Score calculation completed."
echo "All steps finished successfully."