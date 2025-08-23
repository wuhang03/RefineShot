import argparse, ast, json, logging, time
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from accelerate import Accelerator
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

HF_MODELS = {
    "InternVL3-2B": "OpenGVLab/InternVL3-2B",
    "InternVL3-8B": "OpenGVLab/InternVL3-8B",
}
DEFAULT_DATA_FILE = "evaluation/data/ShotBench/test.tsv"


def safe_load(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def build_prompt(row: pd.Series, root_dir: Path, default_fps: float):
    q = row["question"]
    options: dict[str, str] = safe_load(row["options"])
    opts_block = "Options:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())

    prompt = (
        f"Question: {q}\n{opts_block}\n"
        "Please select the most likely answer from the options above."
    )

    # ...existing code...
    # prompt = (
    #     f"Question: {q}\n{opts_block}\n"
    #     "Please think step by step and provide your reasoning and answer in the following format:\n"
    #     "<think>\n"
    #     "[Your detailed reasoning process here. Analyze the image/video and compare it with the given options.]\n"
    #     "</think>\n"
    #     "<answer>\n"
    #     "[The single most likely option letter, e.g., A, B, C, or D]\n"
    #     "</answer>\n"
    #     "You must include the option letter in your answer. Do not output more than one <think> or <answer> block."
    # )

    # ...existing code...
    # prompt = (
    #     f"Question: {q}\n{opts_block}\n"
    #     "Please answer in the following format, and output each section only ONCE:\n"
    #     "<think>\n"
    #     "Step 1: Provide a definition or explanation for each option above.\n"
    #     "Step 2: Summarize the key differences among the options and describe how to judge between them.\n"
    #     "Step 3: Analyze the given image based on these differences.\n"
    #     "Step 4: Select the most likely answer from the options, ensuring it is consistent with the reasoning process.\n"
    #     "At the end, clearly state: 'Therefore, the correct answer is X.' "
    #     "You must use the same X in the <answer> section.\n"
    #     "</think>\n"
    #     "<answer>\n"
    #     "X  # The answer here must exactly match the option stated in <think>.\n"
    #     "</answer>\n"
    #     "If the answers in <think> and <answer> do not match, your response will be considered incorrect."
    # )
# ...existing code...

    # prompt = (
    #     # ===== 示例 (demonstration) =====
    #     "Here is an example:\n\n"
    #     "Question: What's the composition of this shot?\n"
    #     "Options:\n"
    #     "A. Balanced, Right heavy\n"
    #     "B. Balanced, Center\n"
    #     "C. Center, Short side\n"
    #     "D. Center, Right heavy\n\n"
    #     "Step 1: Provide a definition or explanation for each option above.\n"
    #     "Step 2: Summarize the key differences among the options and describe how to judge between them.\n"
    #     "Step 3: Analyze the given image based on these differences.\n"
    #     "Step 4: Select the most likely answer from the options, ensuring it is consistent with the reasoning process.\n"
    #     "At the end of <think>, clearly state: 'Therefore, the correct answer is X.'\n"
    #     "Then in <answer>, repeat exactly that X.\n\n"
    #     "=== Demonstration Answer ===\n"
    #     "<think>The image shows a person standing in a room, facing slightly towards the right side of the frame. "
    #     "The composition centers around the individual, with no other significant objects or people visible. "
    #     "The background includes a plant and some furniture, but they do not detract from the main subject. "
    #     "Therefore, the composition is centered and slightly off-center to the right.\n\n"
    #     "Option A: Balanced, Right heavy - This option suggests a balanced composition where the right side is more prominent than the left. "
    #     "However, the image does not show any significant imbalance or emphasis on one side over the other.\n"
    #     "Option B: Balanced, Center - This option implies a balanced composition where both sides are equal in prominence. "
    #     "The image does not show any significant imbalance, so this option could be considered.\n"
    #     "Option C: Center, Short side - This option suggests a composition where the subject is centered and aligned with the short side of the frame. "
    #     "The image does not align the subject with the short side of the frame.\n"
    #     "Option D: Center, Right heavy - This option suggests a composition where the subject is centered and more prominently featured on the right side. "
    #     "The image does not show any significant emphasis on the right side over the center.\n\n"
    #     "Given the analysis, the most appropriate description for the composition is 'Center, Right heavy,' "
    #     "as the subject is centered but more prominently featured on the right side of the frame. "
    #     "Therefore, the correct answer is D.</think>\n"
    #     "<answer>D</answer>\n\n"
        
    #     # ===== 目标任务 (用户输入) =====
    #     "Now solve the following:\n\n"
    #     f"Question: {q}\n{opts_block}\n"
    #     "Step 1: Provide a definition or explanation for each option above.\n"
    #     "Step 2: Summarize the key differences among the options and describe how to judge between them.\n"
    #     "Step 3: Analyze the given image based on these differences.\n"
    #     "Step 4: Select the most likely answer from the options, ensuring it is consistent with the reasoning process.\n"
    #     "In <think>, explain your reasoning in detail. At the end of <think>, clearly state the conclusion once in the form: 'Therefore, the correct answer is X.'\n"
    #     "Do not repeat this conclusion multiple times.\n"
    #     "Close the reasoning with </think>.\n"
    #     "Then in <answer>, output only the letter X without any extra text, and close with </answer>."

    # )


    types = safe_load(row["type"])
    paths = safe_load(row["path"])
    msgs: List[dict] = []
    for t, rel_path in zip(types, paths):
        abs_path = (root_dir / rel_path).resolve()
        if t == "image":
            msgs.append({"type": "image", "image": str(abs_path)})
        elif t == "video":
            msgs.append(
                {"type": "video", "video": str(abs_path), "max_pixels": 360 * 640, "fps": default_fps}
            )
        else:
            raise ValueError(f"Unsupported modality type: {t}")
    return msgs, prompt


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate InternVL on ShotBench (multi-GPU)")
    p.add_argument("--model", choices=HF_MODELS.keys(), default="InternVL3-8B")
    p.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="TSV file path")
    p.add_argument("--root-dir", default=None)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--category", default="composition")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    p.add_argument("--session-len", type=int, default=16384, help="Session length")
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    
    repo_id = HF_MODELS[args.model]
    root_dir = Path(args.root_dir) if args.root_dir else Path(args.data_file).parent

    # 设置输出目录和日志
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    
    world_size, rank = accelerator.num_processes, accelerator.process_index
    
    # 添加 logging 配置
    ts = int(time.time())
    logging.basicConfig(
        filename=out_dir / f"run_{ts}_rank{rank}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    accelerator.print(f"🔄 Loading InternVL from {repo_id} using lmdeploy...")
    logging.info(f"Loading InternVL from {repo_id} using lmdeploy...")
    
    # 使用 lmdeploy 初始化管道
    backend_config = TurbomindEngineConfig(
        session_len=args.session_len,
        tp=args.tp
    )
    chat_template_config = ChatTemplateConfig(model_name='internvl2_5')
    
    pipe = pipeline(
        repo_id,
        backend_config=backend_config,
        chat_template_config=chat_template_config
    )

    full_df = pd.read_csv(args.data_file, sep="\t")
    local_df = full_df.iloc[rank::world_size].reset_index(drop=True)
    local_df["prediction"] = ""

    logging.info(f"Rank {rank}/{world_size}: Processing {len(local_df)} samples")

    # 推理循环
    for idx, row in tqdm(local_df.iterrows(), total=len(local_df)):
        try:
            # if row["category"] != args.category: 
            #     local_df.at[idx, "prediction"] = "Z"
            #     continue
                
            vision_msgs, prompt = build_prompt(row, root_dir, args.fps)

            # logging.info(f"Processing row {idx}")
            print("vision_msgs: ", vision_msgs)
            print("prompt: ", prompt)
            
            # 处理图像输入
            images = []
            for msg in vision_msgs:
                if msg["type"] == "image":
                    img_path = msg["image"]
                    # 检查文件是否存在
                    if not Path(img_path).exists():
                        accelerator.print(f"Warning: Image file not found: {img_path}")
                        logging.warning(f"Image file not found: {img_path}")
                        continue
                    
                    # 使用 lmdeploy 的 load_image 加载图像
                    image = load_image(img_path)
                    images.append(image)
            
            # 使用 lmdeploy 管道进行推理
            if images:
                # 对于单张图片
                if len(images) == 1:
                    gen_config = {
                        'max_new_tokens': args.max_new_tokens,
                        'temperature': 0.0,
                        'do_sample': False
                    }

                    response = pipe((prompt, images[0]), **gen_config)
                else:
                    # 对于多张图片，可能需要调整prompt格式
                    response = pipe((prompt, images), **gen_config)
                
                prediction = response.text if hasattr(response, 'text') else str(response)
                local_df.at[idx, "prediction"] = prediction
                accelerator.print(f"Row {idx}: {prediction}")
                logging.info(f"Row {idx} completed: {prediction}")  # 完整记录
                print("\n\n=== Predicted Answer ===")
                print(prediction)
                print("=== End ===\n\n")
            
        except Exception as e:
            accelerator.print(f"Error processing row {idx}: {e}")
            logging.error(f"Error processing row {idx}: {e}")
            local_df.at[idx, "prediction"] = ""

    # 保存结果
    output_file = out_dir / f"rank_{rank}_predictions.xlsx"
    local_df.to_excel(output_file, index=False)
    accelerator.print(f"✅ Saved predictions to {output_file}")
    logging.info(f"Saved predictions to {output_file}")

    # 如果需要合并多个进程的结果
    if accelerator.is_main_process:
        # 等待所有进程完成
        accelerator.wait_for_everyone()
        
        # 合并所有进程的结果
        all_files = list(out_dir.glob("rank_*_predictions.xlsx"))  # 修正文件扩展名
        if all_files:
            all_dfs = []
            for file in all_files:
                df = pd.read_excel(file)  # 使用 read_excel
                all_dfs.append(df)
            
            merged_df = pd.concat(all_dfs, ignore_index=True).sort_index()
            final_output = out_dir / f"predictions_merged.xlsx"
            merged_df.to_excel(final_output, index=False)
            accelerator.print(f"✅ Merged results saved to {final_output}")
            logging.info(f"Merged results saved to {final_output}")


if __name__ == "__main__":
    main()