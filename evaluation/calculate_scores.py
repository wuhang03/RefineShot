import argparse, json, os, random, re, string, time
from pathlib import Path
import copy as cp
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import datetime  

parser = argparse.ArgumentParser()
parser.add_argument("--prediction_path", required=True)
parser.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY"))
parser.add_argument("--model", default="gpt-4o-2024-08-06")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--category", default="composition")
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

def eval_row(row, gpt_model):
    choices = build_choices(row)
    opts_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
    cost = 0

    print("\n\n=== Predicted Answer ===")
    print(row["prediction"])
    pred_letter = can_infer(row["prediction"], choices)
    print("=== End ===\n\n")

    print("\n\n=== Predicted Letter ===")
    print(pred_letter)
    print("=== End ===\n\n")
    # if not pred_letter and gpt_model:
    #     prompt = build_prompt(row["question"], opts_str, row["prediction"])
    #     for _ in range(3):
    #         ans, cost = openai_generate(prompt, gpt_model, temperature=args.temperature)
    #         print("ans: ", ans)
    #         print("cost: ", cost)
    #         pred_letter = can_infer(ans, choices)
    #         if pred_letter:
    #             break
    #     if not pred_letter:
    #         pred_letter = random.choice(list(choices) + ["Z"])
    # else:
    #     pred_letter = "Z"

    if not pred_letter:
        pred_letter = 'Z'

    
    hit = int(pred_letter == row["answer"])
    return hit, pred_letter or "Z", cost

def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_excel(path, sheet_name=0)

# def read_input(p: str):
#     pth = Path(p)
#     if pth.is_file():
#         return load_df(str(pth)), str(pth)
#     elif pth.is_dir():
#         frames = []
#         for fn in pth.iterdir():
#             if fn.suffix.lower() in (".xlsx", ".xls", ".tsv"):
#                 frames.append(load_df(str(fn)))
#         if not frames:
#             raise ValueError("Prediction File Not Found!")
#         return pd.concat(frames, ignore_index=True), str(pth / "merged_eval.xlsx")
#     raise FileNotFoundError(p)

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
    df, out_path = read_input(PRED_PATH)
    hits, letters = [], []
    total_cost = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["category"] != args.category:
            hit = 0
            letter = 'O'
            cost = 0
        else :
            hit, letter, cost = eval_row(row, args.model)
        hits.append(hit)
        letters.append(letter)
        total_cost += cost

    print("total_cost:", total_cost)
    df["pred_letter"] = letters
    df["hit"] = hits

    grp = df.groupby("category")["hit"].agg(total="count", correct="sum").reset_index()
    grp["accuracy"] = grp["correct"] / grp["total"]
    overall = pd.DataFrame({
        "category": ["Overall"],
        "total": [len(df)],
        "correct": [sum(hits)],
        "accuracy": [sum(hits) / len(df)]
    })
    acc_df = pd.concat([grp, overall], ignore_index=True)
    print(acc_df)

    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as w:
        df.to_excel(w, index=False, sheet_name="Results")
        acc_df.to_excel(w, index=False, sheet_name="Accuracy")
    print(f"✅ Write scores to {out_path}")

if __name__ == "__main__":
    main()
