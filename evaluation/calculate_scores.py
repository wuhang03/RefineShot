import argparse, json, os, random, re, string, time
from pathlib import Path
import copy as cp
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--prediction_path", required=True)
parser.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY"))
parser.add_argument("--model", default="gpt-4o-2024-08-06")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--category", default="composition")
parser.add_argument("--check_consistency", action="store_true")
args = parser.parse_args()
PRED_PATH = os.path.abspath(args.prediction_path)

client = OpenAI(api_key=args.openai_key)

# def openai_generate(prompt, model, temperature=0.0):
#     return client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model=model,
#         temperature=temperature,
#     ).choices[0].message.content
def openai_generate(prompt, model, temperature=0.0):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
    )
    
    content = response.choices[0].message.content
    usage = response.usage  # 包含 prompt_tokens, completion_tokens, total_tokens

    # 获取 token 数量
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # 价格计算（GPT-4o 单价）
    input_price_per_1k = 0.005
    output_price_per_1k = 0.015
    cost = (prompt_tokens / 1000) * input_price_per_1k + (completion_tokens / 1000) * output_price_per_1k

    print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total: {total_tokens}")
    print(f"Estimated cost: ${cost:.6f}")

    return response, cost

def extract_from_answer_tag(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    return m.group(1).strip() if m else text.strip()

def build_choices(item) -> dict:
    opts = json.loads(item["options"])
    return {k: v for k, v in opts.items() if v and not pd.isna(v)}

def can_infer_option(answer: str, choices: dict) -> str | bool:
    answer = extract_from_answer_tag(answer)
    answer_mod = cp.copy(answer)
    for c in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(c, " ")
    splits = [s.strip() for s in answer_mod.split()]
    hits = [k for k in choices if k in splits]
    if len(hits) == 1:
        return hits[0]
    if not hits and any(z in splits for z in ("Z", "")):
        return "Z"
    return False

def can_infer_text(answer: str, choices: dict) -> str | bool:
    raw = extract_from_answer_tag(answer).lower()
    hits = [k for k, v in choices.items() if str(v).lower() in raw]
    return hits[0] if len(hits) == 1 else False

def can_infer(answer: str, choices: dict) -> str | bool:
    return can_infer_option(answer, choices) or can_infer_text(answer, choices)

def build_prompt(question, options, prediction):
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several "
        "options of a single-choice question. You are provided with a question, "
        "several options, and an answer, and you need to find which option is most "
        "similar to the answer. If the answer says things like refuse to answer, "
        "I'm sorry cannot help, etc., output Z. If the meaning of all options are "
        "significantly different from the answer, or the answer does not select "
        "any option, output Z. You should output one of the choices, A, B, C, D "
        "(if they are valid options), or Z.\n"
        "Example 1:\nQuestion: Which point is closer to the camera?\nSelect from "
        "the following choices.\nOptions: A. Point A\nB. Point B\n(Z) Failed\n"
        "Answer: Point B, where the child is sitting, is closer to the camera.\n"
        "Your output: (B)\n"
        "Example 2:\nQuestion: Which point is closer to the camera?\nSelect from "
        "the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\n"
        "Answer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
        "Example 3:\nQuestion: Which point is corresponding to the reference point?\n"
        "Select from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\n"
        "Answer: ...\nYour output: (Z)\n"
        "Example 4:\nQuestion: {}?\nOptions: {}\n(Z) Failed\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)

def check_answer_consistency(question, options, prediction, model, tokenizer):
    """
    使用Qwen3模型检查prediction中<think>和<answer>的一致性
    """
    consistency_prompt = f"""
    Please analyze the following response and extract the answers from both <think> and <answer> sections.

    Question: {question}
    Options: {options}
    Response: {prediction}

    Your task:
    1. Extract the answer concluded in the <think> section:
    - First, look for explicit statements like "Therefore, the correct answer is X"
    - If no explicit answer is found, analyze the reasoning content and determine which option it describes or supports based on the context and logic
    - Match the reasoning with the available options to identify the implied answer
    2. Extract the answer from the <answer> section
    3. Return both answers in this exact format:

    THINK_ANSWER: [X]
    ANSWER_SECTION: [Y]

    Where X is the answer from <think> (either explicit or inferred from reasoning) and Y is the answer from <answer>. If either section doesn't contain a clear answer or cannot be reasonably inferred, use "NONE".

    Please provide only the result in the specified format without any additional explanation or reasoning.
    """

    try:
        # 准备模型输入
        messages = [
            {"role": "user", "content": consistency_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 启用thinking模式
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 解析thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        # print(f"Thinking content: {thinking_content}")
        print(f"Final content: {content}")
        
        # 从content中提取答案
        think_match = re.search(r'THINK_ANSWER:\s*\[?([A-G]|NONE)\]?', content, re.IGNORECASE)
        answer_match = re.search(r'ANSWER_SECTION:\s*\[?([A-G]|NONE)\]?', content, re.IGNORECASE)
        
        think_answer = think_match.group(1).upper() if think_match and think_match.group(1).upper() != "NONE" else None
        answer_section = answer_match.group(1).upper() if answer_match and answer_match.group(1).upper() != "NONE" else None

        # print("Think answer: ", think_answer)
        # print("Answer section: ", answer_section)
        
        if think_answer != None and answer_section != None and think_answer != answer_section:
            print("Inconsistency detected.")
            return False
        else:
            return True
    
    except Exception as e:
        print(f"Error in consistency check: {e}")
        return True  # 出错时认为一致

def eval_row(row, gpt_model, evaluator, tokenizer):
    choices = build_choices(row)
    opts_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
    cost = 0

    if pd.isna(row["prediction"]) or str(row["prediction"]).lower() == "nan":
        row["prediction"] = "Z"

    print("prediction: ", row["prediction"])
    
    # 检查一致性
    if args.check_consistency:
        is_consistent = check_answer_consistency(row["question"], opts_str, row["prediction"], evaluator, tokenizer)
        if not is_consistent:
            pred_letter = 'Z'
        else:
            pred_letter = can_infer(row["prediction"], choices)
    else:
        pred_letter = can_infer(row["prediction"], choices)

    print("\n=== Predicted Letter ===")
    print(pred_letter)
    print("=== End ===\n\n")

    if not pred_letter:
        pred_letter = 'Z'

    hit = int(pred_letter == row["answer"])
    return hit, pred_letter or "Z", cost

def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_excel(path, sheet_name=0)

def read_input(p: str):
    pth = Path(p)
    if pth.is_file():
        return load_df(str(pth)), str(pth)
    elif pth.is_dir():
        frames = []
        for fn in pth.iterdir():
            if fn.suffix.lower() in (".xlsx", ".xls", ".tsv"):
                frames.append(load_df(str(fn)))
        if not frames:
            raise ValueError("Prediction File Not Found!")
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"merged_eval_{now_str}.xlsx"
        return pd.concat(frames, ignore_index=True), str(pth / out_name)
    raise FileNotFoundError(p)

def main():
    # if args.check_consistency:
    print("Loading Qwen3 model for consistency checking...")
    model_name = "Qwen/Qwen3-4B"
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    evaluator = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    evaluator.eval()

    df, out_path = read_input(PRED_PATH)
    hits, letters = [], []
    total_cost = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["category"] != args.category and args.category != "all":
            hit = 0
            letter = 'Z'
            cost = 0
        else:
            hit, letter, cost = eval_row(row, args.model, evaluator, tokenizer)
            
        hits.append(hit)
        letters.append(letter)
        total_cost += cost

    print("total_cost:", total_cost)
    df["pred_letter"] = letters
    df["hit"] = hits

    grp = df.groupby("category")["hit"].agg(total="count", correct="sum").reset_index()
    grp["accuracy"] = (grp["correct"] / grp["total"]).round(4) 
    overall = pd.DataFrame({
        "category": ["Overall"],
        "total": [len(df)],
        "correct": [sum(hits)],
        "accuracy": [round(sum(hits) / len(df), 4)]
    })
    acc_df = pd.concat([grp, overall], ignore_index=True)
    print(acc_df)
    
    out_path = out_path.replace('.xlsx', '_results.xlsx')
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as w:
        df.to_excel(w, index=False, sheet_name="Results")
        acc_df.to_excel(w, index=False, sheet_name="Accuracy")
    print(f"✅ Write scores to {out_path}")

if __name__ == "__main__":
    main()