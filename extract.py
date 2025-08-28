import pandas as pd
import json
import ast

def extract_answers_from_tsv():
    try:
        # 读取TSV文件
        df = pd.read_csv('test_origin.tsv', sep='\t', encoding='utf-8')
        
        # 检查是否存在answer列和options列
        if 'answer' not in df.columns:
            print("错误：文件中没有找到'answer'列")
            return
        
        if 'options' not in df.columns:
            print("错误：文件中没有找到'options'列")
            return
            
        if 'category' not in df.columns:
            print("错误：文件中没有找到'category'列")
            return
        
        # 生成问题列表
        questions = []
        
        for index, row in df.iterrows():
            answer_key = str(row['answer']).strip()  # 答案键（如A, B, C, D）
            options_str = str(row['options'])  # options字符串
            category = str(row['category']).strip()  # 类别
            
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
                    
                    # 生成问题
                    question = f"Is the {category} of this shot '{answer_text}'?"
                    questions.append(question)
                    
                    print(f"答案 {answer_key}: {answer_text}")
                    print(f"生成问题: {question}")
                    print("-" * 50)
                else:
                    print(f"警告：在选项中未找到答案键 '{answer_key}'")
                    question = f"Is the {category} of this shot {answer_key}?"
                    questions.append(question)
                    
            except Exception as e:
                print(f"解析选项时出错 (行 {index+1}): {e}")
                print(f"原始选项: {options_str}")
                question = f"Is the {category} of this shot {answer_key}?"
                questions.append(question)
        
        # 创建新的DataFrame，包含原始数据和新生成的问题
        df_new = df.copy()
        df_new['question'] = questions
        df_new['options'] = '{"A": "Yes", "B": "No"}'  # 新的选项格式
        df_new['answer'] = 'A'  # 所有答案都是A（Yes）
        
        # 保存到新的TSV文件
        df_new.to_csv('test.tsv', sep='\t', index=False, encoding='utf-8')
        
        print(f"\n成功生成了 {len(questions)} 个问题并保存到 test_classification.tsv")
        print("新文件包含以下列：")
        print(df_new.columns.tolist())
        
    except FileNotFoundError:
        print("错误：未找到test.tsv文件")
    except Exception as e:
        print(f"处理过程中出现错误：{e}")

# 运行代码
if __name__ == "__main__":
    extract_answers_from_tsv()