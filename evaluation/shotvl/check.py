if "<think>" in answer and "<answer>" in answer:
    # 构建一致性检查和修正的prompt
    consistency_prompt = f"""
Please carefully review the following response and check if the reasoning in <think> and the final answer in <answer> are consistent.

{answer}

Your task:
1. Extract the conclusion from the <think> section (look for statements like "Therefore, the correct answer is X")
2. Extract the answer from the <answer> section
3. Check if they match exactly
4. If they are consistent, respond with: CONSISTENT
5. If they are inconsistent, respond in this exact format:
   INCONSISTENT
   THINK_ANSWER: [X]
   ANSWER_SECTION: [Y]
   where X is the answer concluded in <think> and Y is the answer in <answer>

Please respond with one of these formats:
- If consistent: CONSISTENT
- If inconsistent: 
  INCONSISTENT
  THINK_ANSWER: [X]
  ANSWER_SECTION: [Y]
"""

    # 创建一致性检查的chat
    consistency_chat = [
        {"role": "system", "content": "You are a helpful assistant that checks reasoning consistency and extracts answers when inconsistent."},
        {"role": "user", "content": [{"type": "text", "text": consistency_prompt}]},
    ]
    
    consistency_text_in = processor.apply_chat_template(
        consistency_chat, tokenize=False, add_generation_prompt=True
    )
    
    consistency_inputs = processor(
        text=[consistency_text_in],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(accelerator.device)
    
    # 生成一致性检查结果
    consistency_gen = gen_model.generate(
        **consistency_inputs,
        max_new_tokens=200,  # 减少token数量，只需要返回答案标识
        do_sample=False,
    )
    consistency_trimmed = consistency_gen[0][consistency_inputs.input_ids.shape[-1]:]
    consistency_result = processor.decode(
        consistency_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print(f"\n=== Consistency Check Result ===")
    print(f"Consistency Result: {consistency_result}")
    
    # 如果不一致，提取并显示两个答案
    if "INCONSISTENT" in consistency_result.upper():
        import re
        
        # 提取think中的答案
        think_match = re.search(r'THINK_ANSWER:\s*\[?([A-G])\]?', consistency_result, re.IGNORECASE)
        # 提取answer部分的答案
        answer_match = re.search(r'ANSWER_SECTION:\s*\[?([A-G])\]?', consistency_result, re.IGNORECASE)
        
        if think_match and answer_match:
            think_answer = think_match.group(1)
            answer_section = answer_match.group(1)
            
            print(f"=== Inconsistency Detected ===")
            print(f"Answer in <think> section: {think_answer}")
            print(f"Answer in <answer> section: {answer_section}")
            
            # 可以选择使用think中的答案作为最终答案
            print(f"Using answer from <think> section: {think_answer}")
            
            # 替换answer部分
            answer_start = answer.find("<answer>")
            answer_end = answer.find("</answer>") + 9
            
            if answer_start != -1 and answer_end != -1:
                new_answer_section = f"<answer>\n{think_answer}\n</answer>"
                answer = answer[:answer_start] + new_answer_section + answer[answer_end:]
                print(f"=== Answer Corrected ===")
            else:
                print("Could not locate <answer> tags for correction")
        else:
            print("Could not extract both answers from consistency check")
            print(f"Raw consistency result: {consistency_result}")
    else:
        print("=== Answer is consistent ===")
    
    # 清理GPU内存
    torch.cuda.empty_cache()