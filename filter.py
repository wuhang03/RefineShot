import pandas as pd
import json
import ast

def extract_answers_from_tsv():
    try:
        # 读取TSV文件
        df = pd.read_csv('test.tsv', sep='\t', encoding='utf-8')
        
        # 检查是否存在answer列和options列
        if 'answer' not in df.columns:
            print("错误：文件中没有找到'answer'列")
            return
        
        if 'options' not in df.columns:
            print("错误：文件中没有找到'options'列")
            return
        
        # 提取答案
        answers = []
        
        for index, row in df.iterrows():
            answer_key = str(row['answer']).strip()  # 答案键（如A, B, C, D）
            options_str = str(row['options'])  # options字符串
            
            try:
                # 尝试解析JSON格式的options
                # 处理可能的格式问题
                options_str = options_str.replace('""', '"')  # 处理双引号转义
                
                # 尝试用ast.literal_eval解析
                try:
                    options_dict = ast.literal_eval(options_str)
                except:
                    # 如果ast失败，尝试json.loads
                    options_dict = json.loads(options_str)
                
                # 根据answer_key提取对应的答案
                if answer_key in options_dict:
                    answer_text = options_dict[answer_key]
                    answers.append(answer_text)
                    print(f"答案 {answer_key}: {answer_text}")
                else:
                    print(f"警告：在选项中未找到答案键 '{answer_key}'")
                    answers.append(answer_key)
                    
            except Exception as e:
                print(f"解析选项时出错 (行 {index+1}): {e}")
                print(f"原始选项: {options_str}")
                answers.append(answer_key)
        
        # 保存到answer.txt
        with open('answer.txt', 'w', encoding='utf-8') as f:
            for i, answer in enumerate(answers, 1):
                f.write(f"{i}. {answer}\n")
        
        print(f"\n成功提取了 {len(answers)} 个答案并保存到 answer.txt")
        
    except FileNotFoundError:
        print("错误：未找到test.tsv文件")
    except Exception as e:
        print(f"处理过程中出现错误：{e}")

# 运行代码
if __name__ == "__main__":
    extract_answers_from_tsv()