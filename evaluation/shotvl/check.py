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
            
            # 让模型判断哪个答案是正确的
            judge_prompt = f"""
Given the original reasoning and two different answers, please determine which answer is correct based on the image and the reasoning provided.

Original reasoning from <think> section:
{answer[answer.find("<think>"):answer.find("</think>") + 8]}

Two different answers:
Answer A: {think_answer} (from reasoning conclusion)
Answer B: {answer_section} (from answer section)

Please analyze the image again and determine which answer (A or B) is correct based on the reasoning and visual evidence.

Respond with only: A or B
"""
            
            # 创建判断的chat (包含图像信息)
            judge_chat = [
                {"role": "system", "content": "You are a helpful assistant that determines the correct answer based on reasoning and visual evidence."},
                {"role": "user", "content": vision_msgs + [{"type": "text", "text": judge_prompt}]},
            ]
            
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
                max_new_tokens=50,
                do_sample=False,
            )
            judge_trimmed = judge_gen[0][judge_inputs.input_ids.shape[-1]:]
            judge_result = processor.decode(
                judge_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            
            print(f"=== Model Judgment ===")
            print(f"Model choice: {judge_result}")
            
            # 根据模型的判断选择正确答案
            if "A" in judge_result.upper():
                final_answer = think_answer
                print(f"Using answer from <think> section: {think_answer}")
            elif "B" in judge_result.upper():
                final_answer = answer_section
                print(f"Using answer from <answer> section: {answer_section}")
            else:
                # 如果模型回答不明确，默认使用think中的答案
                final_answer = think_answer
                print(f"Model judgment unclear, defaulting to <think> answer: {think_answer}")
            
            # 替换answer部分
            answer_start = answer.find("<answer>")
            answer_end = answer.find("</answer>") + 9
            
            if answer_start != -1 and answer_end != -1:
                new_answer_section = f"<answer>\n{final_answer}\n</answer>"
                answer = answer[:answer_start] + new_answer_section + answer[answer_end:]
                print(f"=== Answer Corrected to: {final_answer} ===")
            else:
                print("Could not locate <answer> tags for correction")
        else:
            print("Could not extract both answers from consistency check")
            print(f"Raw consistency result: {consistency_result}")
    else:
        print("=== Answer is consistent ===")
    
    # 清理GPU内存
    torch.cuda.empty_cache()