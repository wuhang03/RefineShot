import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Comic Neue'

# 新数据
models = ['Qwen2.5VL-3B', 'Qwen2.5VL-7B', 'ShotVL-3B', 'ShotVL-7B']
performance = [47.5, 51.7, 67.8, 70.2]  # Overall
fRS = [98.5, 98.9, 83.2, 93.0]
IAS = [89.1, 94.8, None, 19.7]  # None表示缺失
colors = ['#6a5acd', '#1e90ff', '#ffa500', '#ff4500']
markers = ['o', 's', 'D', '^']
sizes = [600, 600, 600, 600]

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# 第一个子图：Performance vs fRS
ax = axes[0]
for i, model in enumerate(models):
    ax.scatter(performance[i]+0.2, fRS[i]-0.5, marker=markers[i], s=sizes[i]*1.2, color='black', alpha=0.3, zorder=1)
    ax.scatter(performance[i], fRS[i], marker=markers[i], s=sizes[i], color=colors[i], edgecolor='black', linewidth=2, zorder=2, label=models[i])
for i, model in enumerate(models):
    ax.text(performance[i]+0.5, fRS[i], model, fontsize=32, va='center', ha='left', weight='bold')
ax.set_xlabel('Performance (Overall)', fontsize=44, labelpad=15)
ax.set_ylabel('Reliability (fRS)', fontsize=44, labelpad=15)
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)
ax.grid(True, linestyle='--', color='gray', alpha=0.7)
ax.set_title('Performance vs fRS', fontsize=36, pad=20)

# 第二个子图：Performance vs IAS
ax2 = axes[1]
for i, model in enumerate(models):
    if IAS[i] is not None:
        ax2.scatter(performance[i]+0.2, IAS[i]-0.5, marker=markers[i], s=sizes[i]*1.2, color='black', alpha=0.3, zorder=1)
        ax2.scatter(performance[i], IAS[i], marker=markers[i], s=sizes[i], color=colors[i], edgecolor='black', linewidth=2, zorder=2, label=models[i])
for i, model in enumerate(models):
    if IAS[i] is not None:
        ax2.text(performance[i]+0.5, IAS[i], model, fontsize=32, va='center', ha='left', weight='bold')
ax2.set_xlabel('Performance (Overall)', fontsize=44, labelpad=15)
ax2.set_ylabel('Reliability (IAS)', fontsize=44, labelpad=15)
ax2.tick_params(axis='x', labelsize=32)
ax2.tick_params(axis='y', labelsize=32)
ax2.grid(True, linestyle='--', color='gray', alpha=0.7)
ax2.set_title('Performance vs IAS', fontsize=36, pad=20)

plt.tight_layout()
plt.savefig('dot_compare.pdf', dpi=300, bbox_inches='tight', format='pdf')
# plt.show()