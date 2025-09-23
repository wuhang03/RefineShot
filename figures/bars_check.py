import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Comic Neue'
plt.rcParams['font.weight'] = 'bold'

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

# --------- 散点图 ---------
fig, ax = plt.subplots(figsize=(14, 8))
for i, model in enumerate(models):
    ax.scatter(x[i] - width/2, overall[i], marker=markers[i], s=sizes[i], color=colors[i], edgecolor='black', linewidth=2, label=model if i==0 else "")
for i, model in enumerate(models):
    ax.scatter(x[i] + width/2, overall_check[i], marker=marker_check[i], s=sizes[i], color=colors[i], edgecolor='black', linewidth=2, label=model+'+check' if i==0 else "")
for i in range(len(models)):
    ax.plot([x[i] - width/2, x[i] + width/2], [overall[i], overall_check[i]], color=colors[i], linestyle='--', linewidth=2)
for i in range(len(models)):
    ax.text(x[i] - width/2, overall[i]+0.7, f'{overall[i]:.1f}', fontsize=32, ha='center', va='bottom', weight='bold')
    ax.text(x[i] + width/2, overall_check[i]-0.7, f'{overall_check[i]:.1f}', fontsize=32, ha='center', va='top', weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=38, fontweight='bold')
ax.set_ylabel('Performance (Overall)', fontsize=44, labelpad=15, fontweight='bold')
ax.tick_params(axis='y', labelsize=32)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
ax.set_ylim(40, 75)
ax.grid(True, linestyle='--', color='gray', alpha=0.7)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker=markers[0], color='w', label='Origin', markerfacecolor=colors[0], markeredgecolor='black', markersize=18, linewidth=0),
    Line2D([0], [0], marker=marker_check[0], color='w', label='+Check', markerfacecolor=colors[0], markeredgecolor='black', markersize=18, linewidth=0)
]
ax.legend(handles=legend_elements, fontsize=32, loc='upper left')
plt.tight_layout()
plt.savefig('bars_check_dot.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close(fig)

# --------- 柱状图 ---------
fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
bar_width = 0.35
bar1 = ax_bar.bar(x - bar_width/2, overall, width=bar_width, color=colors, alpha=0.8, label='Origin', edgecolor='black')
bar2 = ax_bar.bar(x + bar_width/2, overall_check, width=bar_width, color=colors, alpha=0.4, label='+Check', edgecolor='black', hatch='//')
for i in range(len(models)):
    ax_bar.text(x[i] - bar_width/2, overall[i]+0.7, f'{overall[i]:.1f}', fontsize=32, ha='center', va='bottom', weight='bold')
    ax_bar.text(x[i] + bar_width/2, overall_check[i]+0.7, f'{overall_check[i]:.1f}', fontsize=32, ha='center', va='bottom', weight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(models, fontsize=38, fontweight='bold')
ax_bar.set_ylabel('Performance (Overall)', fontsize=44, labelpad=15, fontweight='bold')
ax_bar.tick_params(axis='y', labelsize=32)
for label in ax_bar.get_yticklabels():
    label.set_fontweight('bold')
ax_bar.set_ylim(40, 75)
ax_bar.grid(True, linestyle='--', color='gray', alpha=0.7)
ax_bar.legend(fontsize=32, loc='upper left')
fig_bar.tight_layout()
fig_bar.savefig('bars_check_bar.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close(fig_bar)