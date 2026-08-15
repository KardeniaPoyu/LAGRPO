import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

# Academic plotting style setup
plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid", context="paper")
# 设置中文字体 (必须在 style/theme 设置之后)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

# File paths
log_dir = "logs"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Updated file list including the new LAGRPO B4 run
files = {
    "PPO": "ppo_metrics.csv",
    "GRPO (G=4)": "grpo_G4_metrics.csv",
    "GRPO (G=8)": "grpo_G8_metrics.csv",
    "GRPO (G=16)": "grpo_G16_metrics.csv",
    "LAGRPO (G=8)": "grpo_B4_G8_metrics.csv"
}

dfs = {}
for name, file in files.items():
    path = os.path.join(log_dir, file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Calculate exploration efficiency eta = (Success Rate / Mean Response Length) * 1000
        df['eta'] = (df['success_rate'] / (df['mean_response_length'] + 1e-5)) * 1000
        dfs[name] = df
    else:
        print(f"Warning: {path} not found. Skipping {name}.")

def smooth(y, box_pts):
    """Simple moving average smoothing."""
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    # Pad to maintain length
    pad_len = len(y) - len(y_smooth)
    return np.pad(y_smooth, (pad_len//2, pad_len - pad_len//2), mode='edge')

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Updated color palette to make LAGRPO stand out
colors = {
    'PPO': '#e74c3c',          # Red
    'GRPO (G=4)': '#3498db',   # Blue
    'GRPO (G=8)': '#2ecc71',   # Green
    'GRPO (G=16)': '#9b59b6',  # Purple
    'LAGRPO (G=8)': '#f39c12'  # Vibrant Orange (Highlight)
}

markers = {
    'PPO': 'd', 
    'GRPO (G=4)': 'o', 
    'GRPO (G=8)': 's', 
    'GRPO (G=16)': '^',
    'LAGRPO (G=8)': '*'        # Star marker for the best model
}

linestyles = {
    'PPO': '--', 
    'GRPO (G=4)': '-', 
    'GRPO (G=8)': '-', 
    'GRPO (G=16)': '-',
    'LAGRPO (G=8)': '-'
}

def plot_with_smoothing(ax, x, y, label, color, marker, linestyle, window=5, is_eta=False):
    y_smooth = smooth(y, window)
    
    # Highlight LAGRPO with a thicker line
    linewidth = 3.5 if "LAGRPO" in label else 2.0
    
    ax.plot(x, y_smooth, label=label, color=color, linewidth=linewidth, linestyle=linestyle)
    
    # Add light shaded region for variance
    std = np.std(y - y_smooth)
    ax.fill_between(x, y_smooth - std*0.5, y_smooth + std*0.5, color=color, alpha=0.1)
    
    # Plot markers sparsely
    markevery = max(1, len(x) // 10)
    ax.plot(x, y_smooth, color=color, marker=marker, markersize=10 if "*" in marker else 7, 
            markevery=markevery, linestyle='None')

# 1. Success Rate
for name, df in dfs.items():
    if 'success_rate' in df.columns:
        plot_with_smoothing(axes[0], df['step'], df['success_rate'] * 100, name, colors[name], markers[name], linestyles[name])

axes[0].set_xlabel('更新步数')
axes[0].set_ylabel('成功率 (%)')
axes[0].yaxis.set_major_locator(MaxNLocator(nbins=6))
axes[0].set_title("训练成功率收敛趋势")

# 2. Mean Response Length
for name, df in dfs.items():
    if 'mean_response_length' in df.columns:
        plot_with_smoothing(axes[1], df['step'], df['mean_response_length'], name, colors[name], markers[name], linestyles[name])

axes[1].set_xlabel('更新步数')
axes[1].set_ylabel('平均生成的 Token 数')
axes[1].set_title("回复长度 (Token 使用量)")

# 3. Exploration Efficiency (eta)
for name, df in dfs.items():
    if 'eta' in df.columns:
        plot_with_smoothing(axes[2], df['step'], df['eta'], name, colors[name], markers[name], linestyles[name], window=8, is_eta=True)

axes[2].set_xlabel('更新步数')
axes[2].set_ylabel(r'$\eta$ 指数 (成功率 / 平均长度 $\times$ 1000)')
axes[2].set_title("探索效率分析")

# Unified legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=True, shadow=True)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18, wspace=0.25)
save_path = os.path.join(plot_dir, 'exploration_efficiency_academic_zh.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
print(f"Academic plot saved to {save_path}")
