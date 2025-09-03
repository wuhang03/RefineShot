import pandas as pd
import re

def extract_think_answer(text):
    """从文本中提取<think>部分的答案"""
    if not isinstance(text, str):
        return None
    
    # 提取<think>部分
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if not think_match:
        return None
    
    think_content = think_match.group(1)
    
    # 在think部分查找"Therefore, the correct answer is X"或类似表述
    answer_patterns = [
        r'therefore,?\s+the\s+correct\s+answer\s+is\s+([A-G])',
        r'the\s+answer\s+is\s+([A-G])',
        r'correct\s+answer:\s*([A-G])',
        r'answer:\s*([A-G])',
        r'so\s+the\s+answer\s+is\s+([A-G])',
        r'thus,?\s+([A-G])',
        r'therefore,?\s+([A-G])'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, think_content, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None

def extract_answer_section(text):
    """从文本中提取<answer>部分的答案"""
    if not isinstance(text, str):
        return None
    
    # 提取<answer>部分
    answer_match = re.search(r'<answer>\s*([A-G])\s*</answer>', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    return None

def main():
    # 读取Excel文件
    file_path = "eval_results/ShotVL-3B/predictions_adjust_results.xlsx"
    
    try:
        # 读取Results sheet
        df = pd.read_excel(file_path, sheet_name='Results')
        
        print(f"总共读取了 {len(df)} 行数据")
        
        inconsistent_rows = []
        think_correct_count = 0
        answer_correct_count = 0
        both_wrong_count = 0
        
        # 遍历每一行，检查prediction列
        for idx, row in df.iterrows():
            prediction = row.get('prediction', '')
            actual_answer = row.get('answer', '')
            
            # 提取think和answer部分的答案
            think_answer = extract_think_answer(prediction)
            answer_section = extract_answer_section(prediction)

            # 检查是否不一致
            if think_answer and answer_section and think_answer != answer_section:
                # 检查哪个答案是正确的
                think_is_correct = (think_answer == actual_answer)
                answer_is_correct = (answer_section == actual_answer)
                
                if think_is_correct and not answer_is_correct:
                    think_correct_count += 1
                    correctness = "Think正确，Answer错误"
                elif answer_is_correct and not think_is_correct:
                    answer_correct_count += 1
                    correctness = "Answer正确，Think错误"
                elif think_is_correct and answer_is_correct:
                    # 这种情况理论上不应该出现，因为两个答案不同但都正确
                    correctness = "都正确（异常情况）"
                else:
                    both_wrong_count += 1
                    correctness = "都错误"
                
                inconsistent_rows.append({
                    'index': idx,
                    'think_answer': think_answer,
                    'answer_section': answer_section,
                    'actual_answer': actual_answer,
                    'think_is_correct': think_is_correct,
                    'answer_is_correct': answer_is_correct,
                    'correctness': correctness,
                    'prediction': prediction
                })
        
        print(f"\n发现 {len(inconsistent_rows)} 行think和answer不一致的数据：")
        print("=" * 80)
        
        # 统计结果
        print(f"\n=== 统计结果 ===")
        print(f"Think答案正确的行数: {think_correct_count}")
        print(f"Answer答案正确的行数: {answer_correct_count}")
        print(f"两个答案都错误的行数: {both_wrong_count}")
        print(f"总计不一致行数: {len(inconsistent_rows)}")
        
        if len(inconsistent_rows) > 0:
            print(f"\nThink正确率: {think_correct_count/len(inconsistent_rows)*100:.2f}%")
            print(f"Answer正确率: {answer_correct_count/len(inconsistent_rows)*100:.2f}%")
            print(f"都错误率: {both_wrong_count/len(inconsistent_rows)*100:.2f}%")
        
        print("\n" + "=" * 80)
        
        # 详细显示每个不一致的行
        # for i, row_data in enumerate(inconsistent_rows, 1):
        #     print(f"\n第 {i} 个不一致的行 (原始行号: {row_data['index']}):")
        #     print(f"Think中的答案: {row_data['think_answer']} {'✓' if row_data['think_is_correct'] else '✗'}")
        #     print(f"Answer中的答案: {row_data['answer_section']} {'✓' if row_data['answer_is_correct'] else '✗'}")
        #     print(f"标准答案: {row_data['actual_answer']}")
        #     print(f"正确性: {row_data['correctness']}")
        #     print(f"完整prediction:")
        #     print("-" * 40)
        #     print(row_data['prediction'])
        #     print("=" * 80)
        
        # if len(inconsistent_rows) == 0:
        #     print("所有行的think和answer部分都是一致的！")
            
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    main()