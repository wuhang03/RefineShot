import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Comic Neue'

# 模型及其overall数据
models = ['Qwen2.5VL-3B', 'Qwen2.5VL-7B', 'ShotVL-3B', 'ShotVL-7B']
overall = [47.5, 51.7, 67.8, 70.2]
overall_check = [46.8, 50.9, 58.9, 66.5]

# 颜色和样式
colors = ['#6a5acd', '#1e90ff', '#ffa500', '#ff4500']
markers = ['o', 's', 'D', '^']
marker_check = ['P', 'X', 'v', '*']  # +check用不同marker
sizes = [600, 600, 600, 600]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))

# 原始点
for i, model in enumerate(models):
    ax.scatter(x[i] - width/2, overall[i], marker=markers[i], s=sizes[i], color=colors[i], edgecolor='black', linewidth=2, label=model if i==0 else "")
# +check点
for i, model in enumerate(models):
    ax.scatter(x[i] + width/2, overall_check[i], marker=marker_check[i], s=sizes[i], color=colors[i], edgecolor='black', linewidth=2, label=model+'+check' if i==0 else "")

# 连线
for i in range(len(models)):
    ax.plot([x[i] - width/2, x[i] + width/2], [overall[i], overall_check[i]], color=colors[i], linestyle='--', linewidth=2)

# 标注数值
for i in range(len(models)):
    ax.text(x[i] - width/2, overall[i]+0.7, f'{overall[i]:.1f}', fontsize=32, ha='center', va='bottom', weight='bold')
    ax.text(x[i] + width/2, overall_check[i]-0.7, f'{overall_check[i]:.1f}', fontsize=32, ha='center', va='top', weight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=38)
ax.set_ylabel('Performance (Overall)', fontsize=44, labelpad=15)
ax.tick_params(axis='y', labelsize=32)
ax.set_ylim(40, 75)
ax.grid(True, linestyle='--', color='gray', alpha=0.7)

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker=markers[0], color='w', label='Origin', markerfacecolor=colors[0], markeredgecolor='black', markersize=18, linewidth=0),
    Line2D([0], [0], marker=marker_check[0], color='w', label='+Check', markerfacecolor=colors[0], markeredgecolor='black', markersize=18, linewidth=0)
]
ax.legend(handles=legend_elements, fontsize=32, loc='upper left')

plt.tight_layout()
plt.savefig('bars.pdf', dpi=300, bbox_inches='tight', format='pdf')
# plt.show()