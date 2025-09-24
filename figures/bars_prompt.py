import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Comic Neue'

# 数据
models = [
    'Qwen2.5VL-3B', 'Qwen2.5VL-3B', 'Qwen2.5VL-3B',
    'Qwen2.5VL-7B', 'Qwen2.5VL-7B', 'Qwen2.5VL-7B',
    'ShotVL-3B', 'ShotVL-3B', 'ShotVL-3B',
    'ShotVL-7B', 'ShotVL-7B', 'ShotVL-7B'
]
prompts = [
    'Direct', 'Reasoning', 'Step-by-step',
    'Direct', 'Reasoning', 'Step-by-step',
    'Direct', 'Reasoning', 'Step-by-step',
    'Direct', 'Reasoning', 'Step-by-step'
]
overall = [
    47.5, 43.0, 44.0,
    51.7, 50.4, 49.1,
    67.8, 62.2, 15.8,
    70.2, 66.1, 34.0
]

# 颜色和样式
model_list = ['Qwen2.5VL-3B', 'Qwen2.5VL-7B', 'ShotVL-3B', 'ShotVL-7B']
colors = ['#6a5acd', '#1e90ff', '#ffa500', '#ff4500']
prompt_offsets = {'Direct': -0.18, 'Reasoning': 0, 'Step-by-step': 0.18}
custom_hatches = {'Direct': '/', 'Reasoning': '\\', 'Step-by-step': 'x'}

# 横坐标：每个模型下不同prompt
x = np.arange(len(model_list))

fig, axes = plt.subplots(1, 2, figsize=(32, 6))  # 降低高度
ax = axes[0]

# 柱状图 (第一个子图)
for j, prompt in enumerate(['Direct', 'Reasoning', 'Step-by-step']):
    bar_vals = []
    for i, model in enumerate(model_list):
        idx = i * 3 + j
        bar_vals.append(overall[idx] if overall[idx] is not None else 0)
    bars = ax.bar(x + prompt_offsets[prompt], bar_vals, width=0.18,
                  color=colors, alpha=0.7, edgecolor='black')
    # 设置hatch
    for bar in bars:
        bar.set_hatch(custom_hatches[prompt])
    # 数值标注
    for i, val in enumerate(bar_vals):
        if overall[i*3 + j] is not None:
            ax.text(x[i] + prompt_offsets[prompt], val + 1.2, f'{val:.1f}',
                fontsize=22, ha='center', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(model_list, fontsize=28)
ax.set_ylabel('Performance (Overall)', fontsize=32, labelpad=10)
ax.tick_params(axis='y', labelsize=22)
for label in ax.get_yticklabels():
    pass
ax.set_ylim(0, 80)
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

# 自定义 legend 图案为白色背景
legend_elements_bar = [
    Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Direct'], label='Direct'),
    Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Reasoning'], label='Reasoning'),
    Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Step-by-step'], label='Step-by-step'),
]
leg = ax.legend(handles=legend_elements_bar, fontsize=22, title=None, frameon=True)
leg.get_frame().set_facecolor('white')

# =============================
# 保存独立的柱状图 (第二个图)
# =============================
fig_bar, ax_bar = plt.subplots(figsize=(16, 6))  # 降低高度
for j, prompt in enumerate(['Direct', 'Reasoning', 'Step-by-step']):
    bar_vals = []
    for i, model in enumerate(model_list):
        idx = i * 3 + j
        bar_vals.append(overall[idx] if overall[idx] is not None else 0)
    bars = ax_bar.bar(x + prompt_offsets[prompt], bar_vals, width=0.18,
                      color=colors, alpha=0.7, edgecolor='black')
    for bar in bars:
        bar.set_hatch(custom_hatches[prompt])
    for i, val in enumerate(bar_vals):
        if overall[i*3 + j] is not None:
            ax_bar.text(x[i] + prompt_offsets[prompt], val + 1.2, f'{val:.1f}',
                        fontsize=22, ha='center', va='bottom')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(model_list, fontsize=28)
ax_bar.set_ylabel('Performance (Overall)', fontsize=32, labelpad=10)
ax_bar.tick_params(axis='y', labelsize=22)
for label in ax_bar.get_yticklabels():
    pass
ax_bar.set_ylim(0, 80)
ax_bar.grid(True, linestyle='--', color='gray', alpha=0.5)

# ✅ 自定义 legend 白底
legend_elements_bar2 = [
    Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Direct'], label='Direct'),
    Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Reasoning'], label='Reasoning'),
    Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Step-by-step'], label='Step-by-step'),
]
leg_bar = ax_bar.legend(handles=legend_elements_bar2, fontsize=22,
                        title=None, frameon=True)
leg_bar.get_frame().set_facecolor('white')
for text in leg_bar.get_texts():
    text.set_color('black')

fig_bar.tight_layout()
fig_bar.savefig('prompt_bars.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close(fig_bar)

print("柱状图已保存为 'prompt_bars.pdf'")

# # 散点图
# ax2 = axes[1]
# for i, model in enumerate(model_list):
#     for j, prompt in enumerate(['Direct', 'Reasoning', 'Step-by-step']):
#         idx = i * 3 + j
#         if overall[idx] is not None:
#             ax2.scatter(
#                 x[i] + prompt_offsets[prompt], overall[idx],
#                 marker=prompt_styles[prompt]['marker'],
#                 s=sizes[i],
#                 color=colors[i],
#                 edgecolor='black',
#                 linewidth=2,
#                 label=f'{model} - {prompt}' if i == 0 else ""
#             )
#             ax2.text(
#                 x[i] + prompt_offsets[prompt], overall[idx]+1.2,
#                 f'{overall[idx]:.1f}', fontsize=22, ha='center', va='bottom', weight='bold'
#             )
# # 连线（同模型不同prompt）
# for i, model in enumerate(model_list):
#     yvals = [overall[i*3 + j] for j in range(3)]
#     xvals = [x[i] + prompt_offsets[p] for p in ['Direct', 'Reasoning', 'Step-by-step']]
#     valid = [(xv, yv) for xv, yv in zip(xvals, yvals) if yv is not None]
#     if len(valid) > 1:
#         ax2.plot([v[0] for v in valid], [v[1] for v in valid], color=colors[i], linestyle='-', linewidth=2, alpha=0.5)
# ax2.set_xticks(x)
# ax2.set_xticklabels(model_list, fontsize=28, fontweight='bold')
# ax2.set_ylabel('Performance (Overall)', fontsize=32, labelpad=10, fontweight='bold')
# ax2.tick_params(axis='y', labelsize=22)
# for label in ax2.get_yticklabels():
#     label.set_fontweight('bold')
# ax2.set_ylim(20, 80)  # 散点图y轴从20开始
# ax2.grid(True, linestyle='--', color='gray', alpha=0.5)
# # 去掉标题
# # ax2.set_title('Scatter Chart', fontsize=32, pad=20)

# # 自定义 legend 图案为白色
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Direct'], label='Direct'),
#     Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Reasoning'], label='Reasoning'),
#     Patch(facecolor='white', edgecolor='black', hatch=custom_hatches['Step-by-step'], label='Step-by-step'),
# ]
# leg2 = ax2.legend(handles=legend_elements, fontsize=18, loc='upper left', title='Prompt', title_fontsize=18, frameon=True)
# leg2.get_frame().set_facecolor('white')

# # 保存散点图为独立文件
# fig_dot, ax_dot = plt.subplots(figsize=(16, 6))  # 降低高度
# for i, model in enumerate(model_list):
#     for j, prompt in enumerate(['Direct', 'Reasoning', 'Step-by-step']):
#         idx = i * 3 + j
#         if overall[idx] is not None:
#             ax_dot.scatter(
#                 x[i] + prompt_offsets[prompt], overall[idx],
#                 marker=prompt_styles[prompt]['marker'],
#                 s=sizes[i],
#                 color=colors[i],
#                 edgecolor='black',
#                 linewidth=2,
#                 label=f'{model} - {prompt}' if i == 0 else ""
#             )
#             ax_dot.text(
#                 x[i] + prompt_offsets[prompt], overall[idx]+1.2,
#                 f'{overall[idx]:.1f}', fontsize=22, ha='center', va='bottom', weight='bold'
#             )
# for i, model in enumerate(model_list):
#     yvals = [overall[i*3 + j] for j in range(3)]
#     xvals = [x[i] + prompt_offsets[p] for p in ['Direct', 'Reasoning', 'Step-by-step']]
#     valid = [(xv, yv) for xv, yv in zip(xvals, yvals) if yv is not None]
#     if len(valid) > 1:
#         ax_dot.plot([v[0] for v in valid], [v[1] for v in valid], color=colors[i], linestyle='-', linewidth=2, alpha=0.5)
# ax_dot.set_xticks(x)
# ax_dot.set_xticklabels(model_list, fontsize=28, fontweight='bold')
# ax_dot.set_ylabel('Performance (Overall)', fontsize=32, labelpad=10, fontweight='bold')
# ax_dot.tick_params(axis='y', labelsize=22)
# for label in ax_dot.get_yticklabels():
#     label.set_fontweight('bold')
# ax_dot.set_ylim(20, 80)  # 独立散点图y轴从20开始
# ax_dot.grid(True, linestyle='--', color='gray', alpha=0.5)
#  # 去掉标题
#  # ax_dot.set_title('Scatter Chart', fontsize=32, pad=20)
# leg_dot = ax_dot.legend(handles=legend_elements, fontsize=18, loc='upper left', title='Prompt', title_fontsize=18, frameon=True)
# leg_dot.get_frame().set_facecolor('white')
# fig_dot.tight_layout()
# fig_dot.savefig('prompt_dot.pdf', dpi=300, bbox_inches='tight', format='pdf')
# plt.close(fig_dot)

# plt.tight_layout()
# plt.savefig('bars_prompt.pdf', dpi=300, bbox_inches='tight', format='pdf')
# # plt.show()