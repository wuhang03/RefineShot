import argparse, ast, json, logging, time
from pathlib import Path
from typing import List
import gc
import re
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
    )

    prompt = (
        f"Question: {q}\n{opts_block}\n"
        "Please follow the steps below and select the most likely answer from the options above."
        "Step 1: Analyze the question.\n"
        "Step 2: Consider each option carefully.\n"
        "Step 3: Reason why each option is right or wrong.\n"
        "Step 4: Conclude with the correct answer."
    )

    

    # if row["category"] == "camera movement":
    #     prompt += (
    #         " The options are types of camera movements. "
    #         "Answer based on the CAMERA's motion (not object motion). "
    #         "Use the rule: background direction is opposite to camera movement."
    #     )

    # prompt = (
    #     f"Question: {q}\n{opts_block}\n"
    #     "Please answer in the following format, and output each section only ONCE:\n"
    #     "<think>\n"
    #     "Step 1: Provide a definition or explanation for each option above.\n"
    #     "Step 2: Summarize the key differences among the options and describe how to judge between them.\n"
    #     "Step 3: Analyze the given image based on these differences.\n"
    #     "Step 4: Select the most likely answer from the options, ensuring it is consistent with the reasoning process.\n"
    #     "At the end of <think>, clearly state: 'Therefore, the correct answer is X.'\n"
    #     "</think>\n"
    #     "<answer>\n"
    #     "X  # The answer here must exactly match the option stated in <think>.\n"
    #     "</answer>\n"
    #     "Do not output more than one <think> or <answer> block for this question. "
    #     "If the answers in <think> and <answer> do not match, your response will be considered incorrect."
    # )


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
    p.add_argument("--category", default="composition")
    p.add_argument("--check_consistency", action="store_true", help="Whether to check and correct answer consistency between <think> and <answer>")
    return p.parse_args()

def resolve_inconsistency_with_model(answer, think_answer, answer_section, vision_msgs, row, processor, gen_model, args, accelerator):
    """
    当检测到think和answer不一致时，让模型基于图像重新判断哪个答案正确
    """
    # 构建判断prompt，让模型基于图像选择正确答案并提供理由
    judge_prompt = f"""
Based on the image/video, I need you to determine which answer is correct.

The reasoning process concluded with: {think_answer}
But the final answer section shows: {answer_section}

Question: {row['question']}
Options: {safe_load(row['options'])}

Please analyze the image/video again and determine which answer ({think_answer} or {answer_section}) is correct based on the visual evidence.

Please provide your response in the following format:
<reasoning>
[Explain your analysis of the image/video and why you choose one answer over the other]
</reasoning>
<final_answer>
[Only the letter of the correct answer: A, B, C, D, etc.]
</final_answer>
"""
    
    # 创建判断的chat (包含原始图像/视频)
    judge_chat = [
        {"role": "system", "content": "You are a helpful assistant that determines the correct answer based on visual evidence. Always provide both reasoning and a final answer."},
        {"role": "user", "content": vision_msgs + [{"type": "text", "text": judge_prompt}]},
    ]

    print(f"=== Check Prompt ===")
    print("vision msg: ", vision_msgs)
    print("judge_prompt: ", judge_prompt)
    
    judge_text_in = processor.apply_chat_template(
        judge_chat, tokenize=False, add_generation_prompt=True
    )
    judge_img_in, judge_vid_in = process_vision_info(judge_chat)
    
    judge_inputs = processor(
        text=[judge_text_in],
        images=judge_img_in,
        videos=judge_vid_in,
        fps=args.fps,
        padding=True,
        return_tensors="pt",
    ).to(accelerator.device)
    
    # 生成判断结果
    judge_gen = gen_model.generate(
        **judge_inputs,
        max_new_tokens=200,  # 增加token数量以容纳推理过程
        do_sample=False,
    )
    judge_trimmed = judge_gen[0][judge_inputs.input_ids.shape[-1]:]
    judge_result = processor.decode(
        judge_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    
    print(f"=== Model Judgment (Full Response) ===")
    print(f"Model response: {judge_result}")
    
    # 提取推理部分和最终答案
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', judge_result, re.DOTALL | re.IGNORECASE)
    final_answer_match = re.search(r'<final_answer>\s*([A-G])\s*</final_answer>', judge_result, re.IGNORECASE)
    
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        print(f"=== Model Reasoning ===")
        print(f"Reasoning: {reasoning}")
    else:
        print("=== Could not extract reasoning from model response ===")
    
    # 从最终答案中提取字母
    if final_answer_match:
        final_answer = final_answer_match.group(1).upper()
        print(f"=== Model Final Decision ===")
        print(f"Final decision: {final_answer}")
        
        # 使用模型选择的答案
        answer_start = answer.find("<answer>")
        answer_end = answer.find("</answer>") + 9
        
        if answer_start != -1 and answer_end != -1:
            new_answer_section = f"<answer>\n{final_answer}\n</answer>"
            updated_answer = answer[:answer_start] + new_answer_section + answer[answer_end:]
            print(f"=== Answer Corrected to: {final_answer} (based on model judgment) ===")
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            return updated_answer
        else:
            print("Could not locate <answer> tags for correction")
    else:
        # 如果无法从final_answer标签中提取，尝试从整个回复中提取
        judge_match = re.search(r'([A-G])', judge_result.upper())
        
        if judge_match:
            final_answer = judge_match.group(1)
            print(f"=== Extracted answer from general response ===")
            print(f"Final decision: {final_answer}")
            
            # 使用模型选择的答案
            answer_start = answer.find("<answer>")
            answer_end = answer.find("</answer>") + 9
            
            if answer_start != -1 and answer_end != -1:
                new_answer_section = f"<answer>\n{final_answer}\n</answer>"
                updated_answer = answer[:answer_start] + new_answer_section + answer[answer_end:]
                print(f"=== Answer Corrected to: {final_answer} (based on model judgment) ===")
                
                # 清理GPU内存
                torch.cuda.empty_cache()
                return updated_answer
            else:
                print("Could not locate <answer> tags for correction")
        else:
            # 如果无法从模型判断中提取答案，默认使用think中的答案
            print(f"=== Could not parse model judgment, using think answer: {think_answer} ===")
    
    # 回退到使用think中的答案
    answer_start = answer.find("<answer>")
    answer_end = answer.find("</answer>") + 9
    
    if answer_start != -1 and answer_end != -1:
        new_answer_section = f"<answer>\n{think_answer}\n</answer>"
        updated_answer = answer[:answer_start] + new_answer_section + answer[answer_end:]
        print(f"=== Answer Corrected to: {think_answer} ===")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        return updated_answer
    else:
        print("Could not locate <answer> tags for correction")
        return answer

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
            if row["category"] != args.category and args.category != "all": 
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


            # inputs = processor(
            #     text=[text_in],
            #     images=img_in,
            #     videos=vid_in,
            #     fps=args.fps,
            #     padding=True,
            #     return_tensors="pt",
            # ).to(accelerator.device)

            # gen = gen_model.generate(
            #     **inputs,
            #     max_new_tokens=args.max_new_tokens,
            #     do_sample=False,
            # )
            # trimmed = gen[0][inputs.input_ids.shape[-1]:]
            # answer = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # print("\n\n=== Predicted Answer ===")
            # print(answer)
            # print("=== End ===\n\n")


            # local_df.at[idx, "prediction"] = answer
            # logging.info(answer)
            # torch.cuda.empty_cache()
            # gc.collect()
            # ...existing code...

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

            print("\n\n=== Initial Predicted Answer ===")
            print(answer)
            print("=== End Initial Answer ===\n\n")
            # 一致性检查和修正
            if args.check_consistency:
                
                def extract_think_answer(text, processor, gen_model, accelerator):
                    """使用模型从文本中提取<think>部分的答案"""
                    # 提取<think>部分
                    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
                    if not think_match:
                        return None
                    
                    think_content = think_match.group(1)
                    
                    # 构建提取答案的prompt
                    extract_prompt = f"""
                Please extract the final answer from the following reasoning text. Look for statements that conclude with a specific answer choice (like A, B, C, D, etc.).

                Reasoning text:
                {think_content}

                Please respond with only the letter of the answer (A, B, C, D, etc.) that this reasoning concludes with. If no clear answer is stated, respond with "NONE".
                """
                    
                    # 创建提取答案的chat
                    extract_chat = [
                        {"role": "system", "content": "You are a helpful assistant that extracts answers from reasoning text."},
                        {"role": "user", "content": [{"type": "text", "text": extract_prompt}]},
                    ]
                    
                    extract_text_in = processor.apply_chat_template(
                        extract_chat, tokenize=False, add_generation_prompt=True
                    )
                    
                    extract_inputs = processor(
                        text=[extract_text_in],
                        images=None,
                        videos=None,
                        padding=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    # 生成提取结果
                    extract_gen = gen_model.generate(
                        **extract_inputs,
                        max_new_tokens=50,
                        do_sample=False,
                    )
                    extract_trimmed = extract_gen[0][extract_inputs.input_ids.shape[-1]:]
                    extract_result = processor.decode(
                        extract_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).strip()
                    
                    print(f"Model extracted from think: {extract_result}")
                    
                    # 从模型回复中提取答案
                    # import re
                    answer_match = re.search(r'([A-G])', extract_result.upper())
                    
                    if answer_match and "NONE" not in extract_result.upper():
                        # 清理GPU内存
                        torch.cuda.empty_cache()
                        return answer_match.group(1).upper()
                    else:
                        # 清理GPU内存
                        torch.cuda.empty_cache()
                        return None
                                
                def extract_answer_section(text):
                    """从文本中提取<answer>部分的答案"""
                    # 提取<answer>部分
                    answer_match = re.search(r'<answer>\s*([A-G])\s*</answer>', text, re.IGNORECASE)
                    if answer_match:
                        return answer_match.group(1).upper()
                    
                    return None
                
                # 提取think和answer部分的答案
                # think_answer = extract_think_answer(answer)
                think_answer = extract_think_answer(answer, processor, gen_model, accelerator)
                answer_section = extract_answer_section(answer)
                
                print(f"\n=== Consistency Check ===")
                print(f"Think答案: {think_answer}")
                print(f"Answer部分答案: {answer_section}")
                
                # 检查一致性
                if think_answer and answer_section:
                    if think_answer == answer_section:
                        print("=== Answer is consistent ===")
                    else:
                        print(f"=== Inconsistency Detected ===")
                        print(f"Think部分结论: {think_answer}")
                        print(f"Answer部分: {answer_section}")
                        print(f"Using answer from <think> section: {think_answer}")
                        
                        # 替换answer部分为think中的答案
                        answer_start = answer.find("<answer>")
                        answer_end = answer.find("</answer>") + 9

                        if answer_start != -1 and answer_end != -1:
                            # 调用函数来解决不一致问题
                            answer = resolve_inconsistency_with_model(
                                answer, think_answer, answer_section, vision_msgs, row,
                                processor, gen_model, args, accelerator
                            )
                        else:
                            print("Could not locate <answer> tags for correction")
                else:
                    if not think_answer:
                        print("警告：无法从<think>部分提取答案")
                    if not answer_section:
                        print("警告：无法从<answer>部分提取答案")
                        
                        # 如果无法提取answer部分，从think中提取答案并重新构建response
                        if think_answer:
                            print(f"从<think>部分提取到答案: {think_answer}，重新构建response")
                            
                            # 提取原始的think部分
                            think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL | re.IGNORECASE)
                            if think_match:
                                think_content = think_match.group(1)
                                
                                # 重新构建完整的response
                                new_response = f"<think>{think_content}</think>\n<answer>\n{think_answer}\n</answer>"
                                answer = new_response
                                print(f"=== 重新构建的response ===")
                                print(answer)
                                print("=== 重新构建完成 ===")
                                
                                # 重新提取answer_section以便后续处理
                                answer_section = think_answer
                                    
                print("=== End Consistency Check ===\n")


            # # 一致性检查和修正
            # if "<think>" in answer and "<answer>" in answer and args.check_consistency:
            #     # 构建一致性检查的prompt
            #     consistency_prompt = f"""
            #         Please carefully review the following response and check if the reasoning in <think> and the final answer in <answer> are consistent.

            #         {answer}

            #         Your task:
            #         1. Extract the conclusion from the <think> section (look for statements like "Therefore, the correct answer is X")
            #         2. Extract the answer from the <answer> section
            #         3. Check if they match exactly
            #         4. If they are consistent, respond with: CONSISTENT
            #         5. If they are inconsistent, respond with: INCONSISTENT - The answer should be [X] based on the reasoning in <think>

            #         Please respond with only one of these formats:
            #         - CONSISTENT
            #         - INCONSISTENT - The answer should be [X] based on the reasoning in <think>
            #         """

            #     # 创建一致性检查的chat
            #     consistency_chat = [
            #         {"role": "system", "content": "You are a helpful assistant that checks reasoning consistency."},
            #         {"role": "user", "content": [{"type": "text", "text": consistency_prompt}]},
            #     ]
                
            #     consistency_text_in = processor.apply_chat_template(
            #         consistency_chat, tokenize=False, add_generation_prompt=True
            #     )
                
            #     consistency_inputs = processor(
            #         text=[consistency_text_in],
            #         images=None,
            #         videos=None,
            #         padding=True,
            #         return_tensors="pt",
            #     ).to(accelerator.device)
                
            #     # 生成一致性检查结果
            #     consistency_gen = gen_model.generate(
            #         **consistency_inputs,
            #         max_new_tokens=100,
            #         do_sample=False,
            #     )
            #     consistency_trimmed = consistency_gen[0][consistency_inputs.input_ids.shape[-1]:]
            #     consistency_result = processor.decode(
            #         consistency_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            #     )
                
            #     print(f"\n=== Consistency Check Result ===")
            #     print(f"Consistency Result: {consistency_result}")
                
            #     # 如果不一致，修正答案
            #     if "INCONSISTENT" in consistency_result.upper():
            #         import re
                    
            #         # 从一致性检查结果中提取正确答案
            #         corrected_match = re.search(r'The answer should be \[?([A-G])\]?', consistency_result, re.IGNORECASE)
                    
            #         if corrected_match:
            #             corrected_answer = corrected_match.group(1)
            #             print(f"Detected inconsistency. Correcting answer to: {corrected_answer}")
                        
            #             # 替换answer部分
            #             answer_start = answer.find("<answer>")
            #             answer_end = answer.find("</answer>") + 9
                        
            #             if answer_start != -1 and answer_end != -1:
            #                 new_answer_section = f"<answer>\n{corrected_answer}\n</answer>"
            #                 answer = answer[:answer_start] + new_answer_section + answer[answer_end:]
            #                 print(f"=== Corrected Answer ===")
            #             else:
            #                 print("Could not locate <answer> tags for correction")
            #         else:
            #             print("Could not extract corrected answer from consistency check")
            #     else:
            #         print("=== Answer is consistent ===")
                
            #     # print("=========================")
                
            #     # 清理GPU内存
            #     torch.cuda.empty_cache()

            print("\n\n=== Final Predicted Answer ===")
            print(answer)
            print("=== End ===\n\n")

            local_df.at[idx, "prediction"] = answer
            print("answer add to excel")
            logging.info(answer)
            torch.cuda.empty_cache()
            gc.collect()

# ...existing code...

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
