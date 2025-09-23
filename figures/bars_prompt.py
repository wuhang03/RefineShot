import matplotlib.pyplot as plt
import numpy as np
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
    67.8, 62.2, None,
    70.2, 66.1, 34.0
]
# 颜色和样式
model_list = ['Qwen2.5VL-3B', 'Qwen2.5VL-7B', 'ShotVL-3B', 'ShotVL-7B']
colors = ['#6a5acd', '#1e90ff', '#ffa500', '#ff4500']
markers = ['o', 's', 'D', '^']
prompt_styles = {'Direct': {'marker': 'o', 'linestyle': '-'},
                 'Reasoning': {'marker': 's', 'linestyle': '--'},
                 'Step-by-step': {'marker': 'D', 'linestyle': ':'}}
sizes = [600, 600, 600, 600]

# 横坐标：每个模型下不同prompt
x = np.arange(len(model_list))
bar_width = 0.18
prompt_offsets = {'Direct': -bar_width, 'Reasoning': 0, 'Step-by-step': bar_width}

fig, axes = plt.subplots(1, 2, figsize=(32, 6))  # 降低高度
ax = axes[0]

custom_hatches = {'Direct': '/', 'Reasoning': '\\', 'Step-by-step': 'x'}

# 柱状图
for j, prompt in enumerate(['Direct', 'Reasoning', 'Step-by-step']):
    bar_vals = []
    for i, model in enumerate(model_list):
        idx = i * 3 + j
        bar_vals.append(overall[idx] if overall[idx] is not None else 0)
    bars = ax.bar(x + prompt_offsets[prompt], bar_vals, width=bar_width, color=colors, alpha=0.7, label=prompt, edgecolor='black')
    # 设置hatch: 同一prompt用同一图案
    for bar in bars:
        bar.set_hatch(custom_hatches[prompt])
    # 数值标注
    for i, val in enumerate(bar_vals):
        if overall[i*3 + j] is not None:
            ax.text(x[i] + prompt_offsets[prompt], val + 1.2, f'{val:.1f}', fontsize=22, ha='center', va='bottom', weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_list, fontsize=28, fontweight='bold')
ax.set_ylabel('Performance (Overall)', fontsize=32, labelpad=10, fontweight='bold')
ax.tick_params(axis='y', labelsize=22)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
ax.set_ylim(0, 80)
ax.grid(True, linestyle='--', color='gray', alpha=0.5)
ax.legend(fontsize=22, title='Prompt', title_fontsize=22)
# 去掉标题
# ax.set_title('Bar Chart', fontsize=32, pad=20)

# 保存柱状图为独立文件
fig_bar, ax_bar = plt.subplots(figsize=(16, 6))  # 降低高度
for j, prompt in enumerate(['Direct', 'Reasoning', 'Step-by-step']):
    bar_vals = []
    for i, model in enumerate(model_list):
        idx = i * 3 + j
        bar_vals.append(overall[idx] if overall[idx] is not None else 0)
    bars = ax_bar.bar(x + prompt_offsets[prompt], bar_vals, width=bar_width, color=colors, alpha=0.7, label=prompt, edgecolor='black')
    for bar in bars:
        bar.set_hatch(custom_hatches[prompt])
    for i, val in enumerate(bar_vals):
        if overall[i*3 + j] is not None:
            ax_bar.text(x[i] + prompt_offsets[prompt], val + 1.2, f'{val:.1f}', fontsize=22, ha='center', va='bottom', weight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(model_list, fontsize=28, fontweight='bold')
ax_bar.set_ylabel('Performance (Overall)', fontsize=32, labelpad=10, fontweight='bold')
ax_bar.tick_params(axis='y', labelsize=22)
for label in ax_bar.get_yticklabels():
    label.set_fontweight('bold')
ax_bar.set_ylim(0, 80)
ax_bar.grid(True, linestyle='--', color='gray', alpha=0.5)
ax_bar.legend(fontsize=22, title='Prompt', title_fontsize=22)
# 去掉标题
# ax_bar.set_title('Bar Chart', fontsize=32, pad=20)
fig_bar.tight_layout()
fig_bar.savefig('prompt_bars.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close(fig_bar)

# 散点图
ax2 = axes[1]
for i, model in enumerate(model_list):
    for j, prompt in enumerate(['Direct', 'Reasoning', 'Step-by-step']):
        idx = i * 3 + j
        if overall[idx] is not None:
            ax2.scatter(
                x[i] + prompt_offsets[prompt], overall[idx],
                marker=prompt_styles[prompt]['marker'],
                s=sizes[i],
                color=colors[i],
                edgecolor='black',
                linewidth=2,
                label=f'{model} - {prompt}' if i == 0 else ""
            )
            ax2.text(
                x[i] + prompt_offsets[prompt], overall[idx]+1.2,
                f'{overall[idx]:.1f}', fontsize=22, ha='center', va='bottom', weight='bold'
            )
# 连线（同模型不同prompt）
for i, model in enumerate(model_list):
    yvals = [overall[i*3 + j] for j in range(3)]
    xvals = [x[i] + prompt_offsets[p] for p in ['Direct', 'Reasoning', 'Step-by-step']]
    valid = [(xv, yv) for xv, yv in zip(xvals, yvals) if yv is not None]
    if len(valid) > 1:
        ax2.plot([v[0] for v in valid], [v[1] for v in valid], color=colors[i], linestyle='-', linewidth=2, alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(model_list, fontsize=28, fontweight='bold')
ax2.set_ylabel('Performance (Overall)', fontsize=32, labelpad=10, fontweight='bold')
ax2.tick_params(axis='y', labelsize=22)
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')
ax2.set_ylim(20, 80)  # 散点图y轴从20开始
ax2.grid(True, linestyle='--', color='gray', alpha=0.5)
# 去掉标题
# ax2.set_title('Scatter Chart', fontsize=32, pad=20)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker=prompt_styles['Direct']['marker'], color='w', label='Direct', markerfacecolor='gray', markeredgecolor='black', markersize=14, linewidth=0),
    Line2D([0], [0], marker=prompt_styles['Reasoning']['marker'], color='w', label='Reasoning', markerfacecolor='gray', markeredgecolor='black', markersize=14, linewidth=0),
    Line2D([0], [0], marker=prompt_styles['Step-by-step']['marker'], color='w', label='Step-by-step', markerfacecolor='gray', markeredgecolor='black', markersize=14, linewidth=0)
]
ax2.legend(handles=legend_elements, fontsize=18, loc='upper left', title='Prompt', title_fontsize=18)

# 保存散点图为独立文件
fig_dot, ax_dot = plt.subplots(figsize=(16, 6))  # 降低高度
for i, model in enumerate(model_list):
    for j, prompt in enumerate(['Direct', 'Reasoning', 'Step-by-step']):
        idx = i * 3 + j
        if overall[idx] is not None:
            ax_dot.scatter(
                x[i] + prompt_offsets[prompt], overall[idx],
                marker=prompt_styles[prompt]['marker'],
                s=sizes[i],
                color=colors[i],
                edgecolor='black',
                linewidth=2,
                label=f'{model} - {prompt}' if i == 0 else ""
            )
            ax_dot.text(
                x[i] + prompt_offsets[prompt], overall[idx]+1.2,
                f'{overall[idx]:.1f}', fontsize=22, ha='center', va='bottom', weight='bold'
            )
for i, model in enumerate(model_list):
    yvals = [overall[i*3 + j] for j in range(3)]
    xvals = [x[i] + prompt_offsets[p] for p in ['Direct', 'Reasoning', 'Step-by-step']]
    valid = [(xv, yv) for xv, yv in zip(xvals, yvals) if yv is not None]
    if len(valid) > 1:
        ax_dot.plot([v[0] for v in valid], [v[1] for v in valid], color=colors[i], linestyle='-', linewidth=2, alpha=0.5)
ax_dot.set_xticks(x)
ax_dot.set_xticklabels(model_list, fontsize=28, fontweight='bold')
ax_dot.set_ylabel('Performance (Overall)', fontsize=32, labelpad=10, fontweight='bold')
ax_dot.tick_params(axis='y', labelsize=22)
for label in ax_dot.get_yticklabels():
    label.set_fontweight('bold')
ax_dot.set_ylim(20, 80)  # 独立散点图y轴从20开始
ax_dot.grid(True, linestyle='--', color='gray', alpha=0.5)
# 去掉标题
# ax_dot.set_title('Scatter Chart', fontsize=32, pad=20)
ax_dot.legend(handles=legend_elements, fontsize=18, loc='upper left', title='Prompt', title_fontsize=18)
fig_dot.tight_layout()
fig_dot.savefig('prompt_dot.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close(fig_dot)

plt.tight_layout()
plt.savefig('bars_prompt.pdf', dpi=300, bbox_inches='tight', format='pdf')
# plt.show()