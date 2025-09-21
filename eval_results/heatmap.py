import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json # 导入 json 库来解析 options 列
import os

# --- 步骤 1: 加载 Excel 文件 (来自您提供的代码) ---
# 您的 Excel 文件名
file_path = 'Qwen2.5-VL-7B/ori_results.xlsx'
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

# --- 步骤 1.5: 读取category.json文件 ---
def load_category_labels():
    """
    从category.json文件中读取类别标签
    """
    category_json_path = 'category.json'
    try:
        with open(category_json_path, 'r', encoding='utf-8') as f:
            category_data = json.load(f)
        print(f"成功读取category.json文件!")
        return category_data
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{category_json_path}'")
        print("请确认category.json文件在同一目录下")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析category.json文件: {e}")
        return None
    except Exception as e:
        print(f"读取category.json时发生错误: {e}")
        return None

# 加载category标签数据
category_labels_data = load_category_labels()

# --- 步骤 2: 处理长格式数据的绘图函数 (修改版) ---
def plot_confusion_matrix_from_long_format(df, category_name, category_labels_data, save_path=None):
    """
    从长格式数据中为指定类别绘制混淆矩阵。
    使用category.json文件中的标签作为轴标签。
    """
    # 1. 根据类别名称筛选数据
    category_df = df[df['category'] == category_name].copy()

    if category_df.empty:
        print(f"警告: 找不到类别 '{category_name}' 的数据，已跳过。")
        return

    # 2. 从category.json中获取该类别的所有标签
    if category_labels_data is None:
        print(f"错误: category.json数据未加载，无法为类别 '{category_name}' 生成图表")
        return
    
    if category_name not in category_labels_data:
        print(f"错误: 在category.json中找不到类别 '{category_name}'")
        print(f"可用的类别有: {list(category_labels_data.keys())}")
        return
    
    # 获取该类别的所有标签
    if 'all' not in category_labels_data[category_name]:
        print(f"错误: 类别 '{category_name}' 中没有'all'字段")
        return
    
    class_labels = category_labels_data[category_name]['all']
    print(f"从category.json中读取到类别 '{category_name}' 的标签: {class_labels}")

    # 3. 逐行解析 'options' 字段并映射 answer 和 pred_letter
    valid_data = category_df.dropna(subset=['answer', 'pred_letter', 'options']).copy()
    if valid_data.empty:
        print(f"警告: 类别 '{category_name}' 没有有效的数据行，已跳过。")
        return

    def map_letter(row, col):
        try:
            letter_map = json.loads(row['options'])
            return letter_map.get(row[col], None)
        except Exception as e:
            print(f"行 options 解析失败: {e}")
            return None

    # 过滤掉 answer 或 pred_letter 对应的标签中含有逗号的行
    def has_comma(row, col):
        try:
            letter_map = json.loads(row['options'])
            label = letter_map.get(row[col], "")
            return "," in label if isinstance(label, str) else False
        except Exception:
            return True  # 解析失败直接过滤

    mask_no_comma = ~(
        valid_data.apply(lambda row: has_comma(row, 'answer'), axis=1) |
        valid_data.apply(lambda row: has_comma(row, 'pred_letter'), axis=1)
    )
    filtered_data = valid_data[mask_no_comma]

    y_true = filtered_data.apply(lambda row: map_letter(row, 'answer'), axis=1)
    y_pred = filtered_data.apply(lambda row: map_letter(row, 'pred_letter'), axis=1)

    # 检查映射是否成功，找出映射失败的数据
    true_unmapped = y_true.isna()
    pred_unmapped = y_pred.isna()

    if true_unmapped.any():
        unmapped_values = filtered_data['answer'][true_unmapped].unique()
        print(f"警告: answer列中有未映射的值: {unmapped_values}")

    if pred_unmapped.any():
        unmapped_values = filtered_data['pred_letter'][pred_unmapped].unique()
        print(f"警告: pred_letter列中有未映射的值: {unmapped_values}")

    # 只保留成功映射的数据
    print(list(y_true))
    valid_mask = ~(y_true.isna() | y_pred.isna())
    y_true_final = y_true[valid_mask]
    y_pred_final = y_pred[valid_mask]
    
    if len(y_true_final) == 0:
        print(f"警告: 类别 '{category_name}' 没有成功映射的数据，已跳过。")
        return
    
    print(f"最终用于混淆矩阵的数据: {len(y_true_final)} 行")

    # 6. 计算混淆矩阵，只使用实际数据中存在的标签
    # 获取实际数据中出现的标签
    print(set(y_true_final))
    print(set(y_pred_final))
    actual_labels_in_data = sorted(list(set(y_true_final) | set(y_pred_final)))
    print(f"实际数据中出现的标签: {actual_labels_in_data}")
    
    # 从category.json的完整标签列表中筛选出实际存在的标签，保持原有顺序
    filtered_class_labels = [label for label in class_labels if label in actual_labels_in_data]
    print(f"用于混淆矩阵的标签: {filtered_class_labels}")
    
    if not filtered_class_labels:
        print(f"警告: 没有找到有效的标签用于生成混淆矩阵")
        return
    
    cm = confusion_matrix(y_true_final, y_pred_final, labels=filtered_class_labels)
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_percent = cm.astype('float') / (cm_sum + 1e-8) * 100

    # 7. 绘图
    plt.figure(figsize=(14, 12))  # 进一步增大图形尺寸
    # 设置更大的字体参数，使用默认字体
    font_kwargs = {'fontsize': 24}
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues',
        xticklabels=filtered_class_labels, 
        yticklabels=filtered_class_labels,
        cbar_kws={'label': 'Prediction Accuracy (%)'},
        annot_kws={'size': 22}
    )
    plt.title(f'Confusion Matrix: {category_name.title()} (%) - {len(y_true_final)} samples', **font_kwargs)
    plt.ylabel('True Label', **font_kwargs)
    plt.xlabel('Predicted Label', **font_kwargs)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.tight_layout()
    
    # 保存图片为pdf
    if save_path:
        # 获取file_path中/之前的第一部分作为前缀
        rel_path = os.path.relpath(file_path, os.path.dirname(__file__))
        prefix = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path.split('/')[0]
        base_filename = os.path.splitext(os.path.basename(save_path))[0]
        save_path_pdf = os.path.join(os.path.dirname(save_path), f"{prefix}_{base_filename}.pdf")
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', format='pdf')
        print(f"图表已保存到: {save_path_pdf}")
    
    plt.show()
    plt.close()  # 关闭图形以释放内存


# --- 步骤 3: 自动识别类别并生成图表 ---
# 检查 DataFrame 是否成功加载
if not main_df.empty and category_labels_data is not None:
    # 指定需要处理的类别
    target_categories = ['shot framing', 'lighting', 'lighting type']
    
    # 从数据中获取实际存在的类别
    all_categories = main_df['category'].dropna().unique()
    print(f"\n在文件中找到以下类别: {all_categories}")
    
    # 筛选出目标类别中实际存在的类别
    categories_to_process = [cat for cat in target_categories if cat in all_categories]
    
    # 进一步检查这些类别是否在category.json中存在
    final_categories = [cat for cat in categories_to_process if cat in category_labels_data]
    
    if not final_categories:
        print(f"\n警告: 目标类别 {target_categories} 在数据或category.json中均不存在！")
        print("请检查类别名称是否正确。")
        print(f"category.json中可用的类别: {list(category_labels_data.keys())}")
    else:
        print(f"\n将处理以下类别: {final_categories}")

        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 为每个找到的类别生成混淆矩阵并保存
        for i, category in enumerate(final_categories):
            print(f"\n--- 正在为 '{category}' 生成图表 ---")
            
            # 为每个类别创建唯一的文件名
            save_filename = f"heatmap_{category.replace(' ', '_').replace('/', '_')}.png"
            save_path = os.path.join(script_dir, save_filename)
            
            plot_confusion_matrix_from_long_format(main_df, category, category_labels_data, save_path)

        print("\n所有图表已生成完毕。")
else:
    if main_df.empty:
        print("\n由于Excel文件加载失败，无法生成任何图表。")
    if category_labels_data is None:
        print("\n由于category.json文件加载失败，无法生成任何图表。")