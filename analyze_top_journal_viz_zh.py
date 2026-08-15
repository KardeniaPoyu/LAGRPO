import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.signal import savgol_filter

# 设置顶刊学术风格 (支持中文)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('ggplot') # Fallback

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.dpi': 600,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def process_data(log_dir="logs"):
    files = {
        "B0: 基准 GRPO": "grpo_ablation_B0_G8_metrics.csv",
        "B1: +长度感知": "grpo_ablation_B1_G8_metrics.csv",
        "B2: +退火机制": "grpo_ablation_B2_G8_metrics.csv",
        "B4: LAGRPO 全功能": "grpo_ablation_B4_FINAL_G8_metrics.csv"
    }
    
    results = {}
    print("正在读取实验日志...")
    for name, filename in files.items():
        path = os.path.join(log_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.head(150) # 为了公平对比，统一截取前 150 步
            df['step'] = range(len(df))
            df['success_rate_pct'] = df['success_rate'] * 100
            df['eta'] = (df['success_rate'] / (df['mean_response_length'] + 1e-5)) * 1000
            results[name] = df
            print(f" - 已加载 {name}: {len(df)} 步")
        else:
            print(f" - 警告: 找不到 {filename}")
    return results

def plot_journal_figure(results, save_path="plots/Fig4_Ablation_Analysis_zh.png"):
    if not results:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    colors = sns.color_palette("Set2", n_colors=len(results))
    markers = ['o', 's', '^', 'D']
    
    # 1. Success Rate Comparison
    ax0 = axes[0]
    for (name, df), color, marker in zip(results.items(), colors, markers):
        y = df['success_rate_pct'].values
        x = df['step'].values
        window = min(11, len(y))
        if window % 2 == 0: window -= 1
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)
        ax0.plot(x, y_smooth, label=name, color=color, linewidth=3)
        ax0.fill_between(x, y_smooth - np.std(y)*0.2, y_smooth + np.std(y)*0.2, color=color, alpha=0.1)
        mark_idx = np.linspace(0, len(x)-1, 10, dtype=int)
        ax0.plot(x[mark_idx], y_smooth[mark_idx], color=color, marker=marker, markersize=9, linestyle='None')

    ax0.set_title("A. 性能收敛曲线", loc='center', fontweight='bold', pad=15)
    ax0.set_ylabel("成功率 (%)", fontsize=15)
    ax0.set_xlabel("训练步数", fontsize=15)
    ax0.set_ylim(0, 50)

    # 2. Token Length Dynamics
    ax1 = axes[1]
    for (name, df), color in zip(results.items(), colors):
        y = df['mean_response_length'].values
        window = min(15, len(y))
        if window % 2 == 0: window -= 1
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)
        ax1.plot(df['step'], y_smooth, color=color, linewidth=3)
        ax1.fill_between(df['step'], y_smooth, y, color=color, alpha=0.1)

    ax1.set_title("B. 推理密度 (Tokens)", loc='center', fontweight='bold', pad=15)
    ax1.set_ylabel("平均回复长度", fontsize=15)
    ax1.set_xlabel("训练步数", fontsize=15)
    ax1.set_ylim(100, 300)

    # 3. Cumulative Exploration Efficiency
    ax2 = axes[2]
    for (name, df), color in zip(results.items(), colors):
        y = df['eta'].values
        window = min(21, len(y))
        if window % 2 == 0: window -= 1
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)
        ax2.plot(df['step'], y_smooth, color=color, linewidth=3, label=name)
        ax2.fill_between(df['step'], y_smooth, y, color=color, alpha=0.1)

    ax2.set_title("C. 探索得分 ($\eta$)", loc='center', fontweight='bold', pad=15)
    ax2.set_ylabel(r"$\eta$ (效率指数)", fontsize=15)
    ax2.set_xlabel("训练步数", fontsize=15)
    ax2.set_ylim(0, 3)

    # Annotation of LAGRPO superiority
    if "B4: LAGRPO 全功能" in results:
        b4_final_eta = results["B4: LAGRPO 全功能"]['eta'].tail(10).mean()
        ax2.annotate(f'LAGRPO 高效平台期', 
                     xy=(140, b4_final_eta), xytext=(40, 2.5),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                     fontsize=12, fontweight='bold')

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), 
               frameon=True, shadow=True, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.25)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    print(f"\n[报告] 最终图表已生成至 {os.path.dirname(save_path)}/")
    print(f" - 高分辨率 PNG: {save_path}")

if __name__ == "__main__":
    data = process_data()
    if data:
        plot_journal_figure(data)
        
        print("\n" + "="*40)
        print("   消融实验总结 (最后一步)")
        print("="*40)
        for name, df in data.items():
            last_sr = df['success_rate_pct'].tail(10).mean()
            last_len = df['mean_response_length'].tail(10).mean()
            last_eta = df['eta'].tail(10).mean()
            print(f"[{name}]")
            print(f"  - 最终成功率 (Top-10 平均): {last_sr:.2f}%")
            print(f"  - 最终 Token 长度 (平均): {last_len:.1f}")
            print(f"  - 探索效率 (eta): {last_eta:.3f}")
        print("="*40)
