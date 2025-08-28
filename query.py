import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

df = pd.read_csv('test_all.tsv', sep='\t')
consistency_list = []

# 遍历每一行，提取 question 和 options，按顺序提问
for idx, row in df.iterrows():
    question = row['question']
    options = row['options']
    print(f"Q{idx+1}: {question}")
    print(f"Options: {options}")
    prompt = (
        f"Question: {question}\n"
        f"Options: {options}\n"
        "Analyze the given options based on relevant professional knowledge. "
        "Determine whether all options measure the same semantic dimension (e.g., all are colors, numbers, object categories, etc.) "
        "AND whether they are mutually exclusive (i.e., at most one option can be correct at the same time). "
        "If both conditions hold, return true. Otherwise, return false."
    )

    prompt = (
        f"Question: {question}\n"
        f"Options: {options}\n"
        "Do these options measure the same dimension, and are they mutually exclusive?"
    )

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1280
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    ans = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    print("\n\n=== Predicted Answer ===")
    print(ans)
    print("=== End ===\n\n")

    consistency_list.append(ans)

# 添加 consistency 列并保存
df['consistency'] = consistency_list
df.to_csv('test_all.tsv', sep='\t', index=False)


