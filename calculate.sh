MODEL_NAME="InternVL3-8B"
MODEL_NAME="Qwen2.5-VL-7B"
# MODEL_NAME="ShotVL-7B"
NUM_GPUS=1
OUTPUT_DIR="eval_results"

CATEGORY="all"

FILENAME="cot.xlsx"

# PREDICTION_PATH="${OUTPUT_DIR}/${MODEL_NAME}/${FILENAME}"

PREDICTION_PATH="${OUTPUT_DIR}/${MODEL_NAME}/${FILENAME}"

OPENAI_API_KEY="" 

echo "Step 2: Calculating scores using OpenAI API"
OPENAI_API_KEY=${OPENAI_API_KEY} python evaluation/calculate_scores.py --prediction_path ${PREDICTION_PATH} --category "${CATEGORY}" --check_adherence

echo "Score calculation completed."
echo "All steps finished successfully.p"