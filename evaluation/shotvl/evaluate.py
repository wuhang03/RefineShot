import argparse, ast, json, logging, time
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration as ShotVLModel
from qwen_vl_utils import process_vision_info

HF_MODELS = {
    "ShotVL-3B": "Vchitect/ShotVL-3B",
    "ShotVL-7B": "Vchitect/ShotVL-7B",
}
DEFAULT_DATA_FILE = "evaluation/data/ShotBench/test.tsv"


def safe_load(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)

# Modified from https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/image_mcq.py
def build_prompt(row: pd.Series, root_dir: Path, default_fps: float):
    q = row["question"]
    options: dict[str, str] = safe_load(row["options"])
    opts_block = "Options:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
    prompt = (
        f"Question: {q}\n{opts_block}\n"
        "Please select the most likely answer from the options above."
        "At the end of <think>, clearly state: 'Therefore, the correct answer is X.'\n"
        "Then in <answer>, repeat exactly that X."
    )

    prompt = (
        f"Question: {q}\n{opts_block}\n"
        "Please answer in the following format, and output each section only ONCE:\n"
        "<think>\n"
        "[Step-by-step reasoning here. At the end, clearly state: 'Therefore, the correct answer is X.' "
        "You must use the same X in the <answer> section.]\n"
        "</think>\n"
        "<answer>\n"
        "X  # The answer here must exactly match the option stated in <think>.\n"
        "</answer>\n"
        "Do not output more than one <think> or <answer> block for this question. "
        "If the answers in <think> and <answer> do not match, your response will be considered incorrect."
    )

    # prompt = (
    #     f"Question: {q}\n{opts_block}\n"
    #     "Step 1: Provide a definition or explanation for each option above.\n"
    #     "Step 2: Summarize the key differences among the options and describe how to judge between them.\n"
    #     "Step 3: Analyze the given image based on these differences.\n"
    #     "Step 4: Select the most likely answer from the options, ensuring it is consistent with the reasoning process."
    #     "At the end of <think>, clearly state: 'Therefore, the correct answer is X.'\n"
    #     "Then in <answer>, repeat exactly that X."
    # )


    # prompt = (
    #     f"Question: {q}\n{opts_block}\n"
    #     "Please think step by step and provide your reasoning and answer "
    #     "in the following format:\n"
    #     "<think>\n"
    #     "Definitions:\n"
    #     "- Option A: ...\n"
    #     "- Option B: ...\n"
    #     "...\n"
    #     "Comparison:\n"
    #     "... (differences)\n"
    #     "Reasoning:\n"
    #     "... (how the shot matches a specific option)\n"
    #     "</think>\n"
    #     "<answer>\n"
    #     "Final Answer: <the chosen option>\n"
    #     "</answer>\n"
    #     "Do not repeat the same content or output multiple <think> sections for one question."
    # )


    types = safe_load(row["type"])
    paths = safe_load(row["path"])
    msgs: List[dict] = []
    for t, rel_path in zip(types, paths):
        abs_path = (root_dir / rel_path).resolve()
        uri = f"file://{abs_path}"
        if t == "image":
            msgs.append({"type": "image", "image": uri})
        elif t == "video":
            msgs.append(
                {"type": "video", "video": uri, "max_pixels": 360 * 640, "fps": default_fps}
            )
        else:
            raise ValueError(f"Unsupported modality type: {t}")
    return msgs, prompt


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ShotVL on ShotBench (multi-GPU)")
    p.add_argument("--model", choices=HF_MODELS.keys(), default="ShotVL-3B")
    p.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="TSV file path")
    p.add_argument("--root-dir", default=None)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--output-dir", default="eval_results")
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    repo_id = HF_MODELS[args.model]
    root_dir = Path(args.root_dir) if args.root_dir else Path(args.data_file).parent

    accelerator.print(f"🔄 Loading {repo_id} ...")
    model = ShotVLModel.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.generation_config.temperature = None
    model = accelerator.prepare(model)

    processor = AutoProcessor.from_pretrained(repo_id)

    full_df = pd.read_csv(args.data_file, sep="\t")
    world_size, rank = accelerator.num_processes, accelerator.process_index
    local_df = full_df.iloc[rank::world_size].reset_index(drop=True)
    local_df["prediction"] = ""

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    logging.basicConfig(
        filename=out_dir / f"run_{ts}_rank{rank}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    system_prompt = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the "
        "answer. The reasoning process and answer are enclosed within <think> "
        "</think> and <answer> </answer> tags."
        if args.reasoning
        else "You are a helpful assistant."
    )

    iterable = tqdm(local_df.iterrows(), total=len(local_df), disable=not accelerator.is_main_process)

    with torch.inference_mode():
        gen_model = accelerator.unwrap_model(model)
        for idx, row in iterable:
            if row["category"] != "composition" : 
                continue
            print("row: ", row)
            # quit()
            vision_msgs, prompt = build_prompt(row, root_dir, args.fps)
            print("vision_msgs: ", vision_msgs)
            print("prompt: ", prompt)

            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": vision_msgs + [{"type": "text", "text": prompt}]},
            ]
            text_in = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(chat)

            # chat = [
            #     {"role": "system", "content": system_prompt},
            #     {"role": "user", "content": prompt},
            # ]
            # text_in = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            # img_in, vid_in = process_vision_info(vision_msgs)


            inputs = processor(
                text=[text_in],
                images=img_in,
                videos=vid_in,
                fps=args.fps,
                padding=True,
                return_tensors="pt",
            ).to(accelerator.device)

            gen = gen_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            trimmed = gen[0][inputs.input_ids.shape[-1]:]
            answer = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            print("-----------------------------------------------------------")
            print("predicted answer: ", answer)

            local_df.at[idx, "prediction"] = answer
            logging.info(answer)

    json_str = local_df.to_json(orient="split", index=False)
    all_json = accelerator.gather_for_metrics([json_str], use_gather_object=True)
    
    if accelerator.is_main_process:
        merged = pd.concat(
            (pd.read_json(s, orient="split") for s in all_json),
            ignore_index=True
        ).sort_index()
        out_path = out_dir / f"predictions_{ts}.xlsx"
        merged.to_excel(out_path, index=False)
        accelerator.print(f"✅ Done! Results saved to {out_path}")

if __name__ == "__main__":
    main()
