import pandas as pd
import json
import ast
import random
import argparse

random.seed(42)

def get_irrelevant_options(num_irrelevant=3):
    """返回指定数量的毫不相关的选项"""
    irrelevant_options = [
        "banana", "purple elephant", "quantum physics", "Tuesday morning",
        "chocolate cake", "flying carpet", "ancient civilization", "digital rainbow",
        "invisible thread", "cosmic dust", "midnight sun", "frozen fire",
        "silent music", "square circle", "liquid stone", "weightless gravity",
        "transparent mirror", "timeless clock", "soundless bell", "motionless dance",
        "colorless painting", "empty fullness", "dark light", "cold fire",
        "dry water", "soft rock", "bitter sweet", "loud silence"
    ]
    return random.sample(irrelevant_options, min(num_irrelevant, len(irrelevant_options)))

def extract_correct_options(num_irrelevant=3):
    try:
        # 读取TSV文件
        df = pd.read_csv('evaluation/data/ShotBench/test.tsv', sep='\t', encoding='utf-8')
        
        # 检查必要的列是否存在
        if 'options' not in df.columns or 'answer' not in df.columns:
            print("Error: Missing 'options' or 'answer' column in the file")
            return
        
        print(f"Total rows found: {len(df)}")
        print(f"Adding {num_irrelevant} irrelevant options to each row")
        print("=" * 80)
        
        # 创建新的DataFrame来保存修改后的数据
        df_modified = df.copy()
        
        for index, row in df.iterrows():
            answer_key = str(row['answer']).strip()
            options_str = str(row['options'])
            
            print(f"\nRow {index + 1}:")
            print(f"Original Answer Key: {answer_key}")
            print(f"Original Options String: {options_str}")
            
            try:
                # 清理选项字符串
                options_str = options_str.replace('""', '"')
                
                # 尝试解析选项字符串
                try:
                    options_dict = ast.literal_eval(options_str)
                except:
                    options_dict = json.loads(options_str)
                
                print(f"Original Parsed Options: {options_dict}")
                
                # 提取正确的选项
                if answer_key in options_dict:
                    correct_option = options_dict[answer_key]
                    print(f"Correct Option: {answer_key} -> {correct_option}")
                    
                    # 获取所有原始选项值
                    original_values = list(options_dict.values())
                    
                    # 添加指定数量的毫不相关的选项
                    irrelevant_values = get_irrelevant_options(num_irrelevant)
                    all_values = original_values + irrelevant_values
                    
                    # 随机打乱所有选项
                    random.shuffle(all_values)
                    
                    # 创建新的选项字典，使用足够的字母作为键
                    total_options = len(all_values)
                    choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
                    
                    if total_options > len(choice_letters):
                        print(f"Warning: Too many options ({total_options}), truncating to {len(choice_letters)}")
                        all_values = all_values[:len(choice_letters)]
                    
                    new_options_dict = {}
                    new_answer_key = None
                    
                    for i, value in enumerate(all_values):
                        if i < len(choice_letters):
                            letter = choice_letters[i]
                            new_options_dict[letter] = value
                            # 找到正确答案的新键
                            if value == correct_option:
                                new_answer_key = letter
                    
                    # 更新DataFrame中的数据
                    new_options_str = json.dumps(new_options_dict, ensure_ascii=False)
                    df_modified.at[index, 'options'] = new_options_str
                    df_modified.at[index, 'answer'] = new_answer_key
                    
                    print(f"New Options: {new_options_dict}")
                    print(f"New Answer Key: {new_answer_key}")
                    print(f"Verification - Correct Option: {new_answer_key} -> {new_options_dict[new_answer_key]}")
                    
                else:
                    print(f"Warning: Answer key '{answer_key}' not found in options")
                    
            except Exception as e:
                print(f"Error parsing options: {e}")
            
            print("-" * 60)
        
        # 根据不相关选项数量生成文件名
        output_filename = f'evaluation/data/ShotBench/test_modified_disturb_{num_irrelevant}.tsv'
        
        # 保存修改后的数据到新文件
        df_modified.to_csv(output_filename, sep='\t', index=False, encoding='utf-8')
        print(f"\nModified data saved to '{output_filename}'")
        print(f"Total rows processed: {len(df)}")
        print(f"Each row now has {len(df.iloc[0]['options'] if len(df) > 0 else [])} options (original + {num_irrelevant} irrelevant)")
    
    except FileNotFoundError:
        print("Error: evaluation/data/ShotBench/test.tsv file not found")
    except Exception as e:
        print(f"Error reading file: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Add irrelevant options to test data')
    parser.add_argument('--num_irrelevant', type=int, default=3,
                       help='Number of irrelevant options to add (default: 3)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_correct_options(args.num_irrelevant)