import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json # 导入 json 库来解析 options 列
import os

# --- 步骤 1: 加载 Excel 文件 (来自您提供的代码) ---
# 您的 Excel 文件名
file_path = 'Qwen2.5-VL-7B/origin_results.xlsx'
# Excel 文件中包含数据的工作表名称
sheet_name = 'Results'

try:
    # 使用 pd.read_excel 读取指定的 Excel 工作表
    main_df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"成功从文件 '{file_path}' 的 '{sheet_name}' 工作表中加载数据!")
    
except FileNotFoundError:
    print(f"错误: 文件未找到!")
    print(f"请确认 '{file_path}' 文件与您的Python脚本在同一个目录下。")
    print("如果不在，请提供文件的完整路径。")
    main_df = pd.DataFrame() # 创建空DataFrame以防后续代码出错
except Exception as e:
    # 捕获其他可能的错误，例如工作表名称不正确
    print(f"读取文件时发生错误: {e}")
    print(f"请检查文件是否损坏，或者名为 '{sheet_name}' 的工作表是否存在。")
    main_df = pd.DataFrame() # 创建空DataFrame以防后续代码出错


# --- 步骤 2: 处理长格式数据的绘图函数 (与之前相同) ---
def plot_confusion_matrix_from_long_format(df, category_name, save_path=None):
    """
    从长格式数据中为指定类别绘制混淆矩阵。
    它会自动从 'options' 列解析标签。
    """
    # 1. 根据类别名称筛选数据
    category_df = df[df['category'] == category_name].copy()

    if category_df.empty:
        print(f"警告: 找不到类别 '{category_name}' 的数据，已跳过。")
        return

    # 2. 解析 'options' 列以获取标签映射
    try:
        options_str = category_df['options'].iloc[0]
        letter_to_label_map = json.loads(options_str)
        class_labels = sorted(letter_to_label_map.values())
    except (json.JSONDecodeError, IndexError) as e:
        print(f"错误: 无法为类别 '{category_name}' 解析 'options' 列: {e}")
        return

    # 3. 提取真实标签和预测标签的字母，并确保数据完整性
    # 先过滤掉包含 NaN 的行
    valid_data = category_df.dropna(subset=['answer', 'pred_letter']).copy()
    
    if valid_data.empty:
        print(f"警告: 类别 '{category_name}' 没有有效的数据行，已跳过。")
        return
    
    y_true_letters = valid_data['answer']
    y_pred_letters = valid_data['pred_letter']
    
    print(f"类别 '{category_name}': 原始数据 {len(category_df)} 行，有效数据 {len(valid_data)} 行")

    # 4. 将字母映射回完整的文本标签
    y_true = y_true_letters.map(letter_to_label_map)
    y_pred = y_pred_letters.map(letter_to_label_map)
    
    # 检查映射是否成功，找出映射失败的数据
    true_unmapped = y_true.isna()
    pred_unmapped = y_pred.isna()
    
    if true_unmapped.any():
        unmapped_values = y_true_letters[true_unmapped].unique()
        print(f"警告: answer列中有未映射的值: {unmapped_values}")
    
    if pred_unmapped.any():
        unmapped_values = y_pred_letters[pred_unmapped].unique()
        print(f"警告: pred_letter列中有未映射的值: {unmapped_values}")
    
    # 只保留成功映射的数据
    valid_mask = ~(y_true.isna() | y_pred.isna())
    y_true_final = y_true[valid_mask]
    y_pred_final = y_pred[valid_mask]
    
    if len(y_true_final) == 0:
        print(f"警告: 类别 '{category_name}' 没有成功映射的数据，已跳过。")
        return
    
    print(f"最终用于混淆矩阵的数据: {len(y_true_final)} 行")

    # 5. 计算混淆矩阵
    cm = confusion_matrix(y_true_final, y_pred_final, labels=class_labels)
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_percent = cm.astype('float') / (cm_sum + 1e-8) * 100

    # 6. 绘图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Prediction Accuracy (%)'})

    plt.title(f'Confusion Matrix: {category_name.title()} (%) - {len(y_true_final)} samples', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    plt.close()  # 关闭图形以释放内存


# --- 步骤 3: 自动识别类别并生成图表 ---
# 检查 DataFrame 是否成功加载
if not main_df.empty:
    # 自动从 'category' 列获取所有唯一的类别
    # 使用 .dropna() 确保我们不处理任何没有类别的行
    all_categories = main_df['category'].dropna().unique()
    print(f"\n在文件中找到以下类别: {all_categories}")

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 为每个找到的类别生成混淆矩阵并保存
    for i, category in enumerate(all_categories):
        print(f"\n--- 正在为 '{category}' 生成图表 ---")
        
        # 为每个类别创建唯一的文件名
        if len(all_categories) > 1:
            save_filename = f"heatmap_{category.replace(' ', '_').replace('/', '_')}.png"
        else:
            save_filename = "heatmap.png"
        
        save_path = os.path.join(script_dir, save_filename)
        
        plot_confusion_matrix_from_long_format(main_df, category, save_path)

    print("\n所有图表已生成完毕。")
else:
    print("\n由于文件加载失败，无法生成任何图表。")