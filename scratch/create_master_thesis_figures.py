import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.signal import savgol_filter

# 设置学术论文风格
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('ggplot')

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'figure.dpi': 600,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

LOG_DIR = "d:/Personal/Documents/GitHub/SLM-RL-Comparation/logs"
PLOT_DIR = "d:/Personal/Documents/GitHub/SLM-RL-Comparation/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def load_log(filename, label, max_steps=180):
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.head(max_steps)
    df['step'] = range(len(df))
    df['sr_pct'] = df['success_rate'] * 100
    df['label'] = label
    # 计算探索效率 eta = SR / Length * 1000
    df['eta'] = (df['success_rate'] / (df['mean_response_length'] + 1e-5)) * 1000
    
    # 平滑处理
    window = min(15, len(df))
    if window % 2 == 0: window -= 1
    if window > 3:
        df['sr_smooth'] = savgol_filter(df['sr_pct'], window, 3)
        df['len_smooth'] = savgol_filter(df['mean_response_length'], window, 3)
        df['eta_smooth'] = savgol_filter(df['eta'], window, 3)
    else:
        df['sr_smooth'] = df['sr_pct']
        df['len_smooth'] = df['mean_response_length']
        df['eta_smooth'] = df['eta']
    return df

def generate_master_fig1_scaling():
    """图 5-1: 算法基准与扩展性分析 (1x2 组合图)"""
    configs = {
        "ppo_metrics.csv": "PPO (Baseline)",
        "grpo_G4_metrics.csv": "GRPO (G=4)",
        "grpo_G8_metrics.csv": "GRPO (G=8)",
        "grpo_G16_metrics.csv": "GRPO (G=16)"
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = sns.color_palette("rocket", n_colors=len(configs))
    
    # (a) Success Rate Convergence
    ax0 = axes[0]
    for i, (fname, label) in enumerate(configs.items()):
        df = load_log(fname, label)
        if df is not None:
            ax0.plot(df['step'], df['sr_smooth'], label=label, color=colors[i], linewidth=2.5)
            ax0.fill_between(df['step'], df['sr_smooth'] - 2, df['sr_smooth'] + 2, color=colors[i], alpha=0.1)
    
    ax0.set_title("(a) Performance Convergence Comparison", fontweight='bold')
    ax0.set_xlabel("Training Steps")
    ax0.set_ylabel("Success Rate (%)")
    ax0.set_ylim(0, 50)
    ax0.legend()

    # (b) Sample Efficiency / Advantage Stability (Using logic from G scaling)
    # 模拟展示组大小与优势方差的关系或直接对比最终表现
    ax1 = axes[1]
    g_values = [4, 8, 16]
    final_sr = []
    for g in g_values:
        df = load_log(f"grpo_G{g}_metrics.csv", f"G={g}")
        if df is not None:
            final_sr.append(df['sr_smooth'].tail(10).mean())
        else:
            final_sr.append(0)
            
    ax1.bar([str(v) for v in g_values], final_sr, color=sns.color_palette("viridis", 3), alpha=0.8)
    ax1.set_title("(b) Final Success Rate vs. Group Size (G)", fontweight='bold')
    ax1.set_xlabel("Group Size (G)")
    ax1.set_ylabel("Final Success Rate (%)")
    ax1.set_ylim(0, 50)
    for i, v in enumerate(final_sr):
        ax1.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "MasterFig1_Scaling_Analysis.svg"))
    plt.savefig(os.path.join(PLOT_DIR, "MasterFig1_Scaling_Analysis.png"), dpi=500)
    print("Generated Master Fig 1 (Scaling/Convergence)")

def generate_master_fig2_ablation():
    """图 5-2: LAGRPO 消融实验综合分析 (1x3 组合图)"""
    configs = {
        "grpo_ablation_B0_G8_metrics.csv": "B0: Baseline GRPO",
        "grpo_ablation_B1_G8_metrics.csv": "B1: +Length Aware",
        "grpo_ablation_B2_G8_metrics.csv": "B2: +Annealing",
        "grpo_ablation_B4_FINAL_G8_metrics.csv": "B4: Full LAGRPO"
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    colors = sns.color_palette("husl", n_colors=len(configs))
    
    labels = list(configs.values())
    
    # (a) Success Rate
    for i, (fname, label) in enumerate(configs.items()):
        df = load_log(fname, label)
        if df is not None:
            axes[0].plot(df['step'], df['sr_smooth'], label=label, color=colors[i], linewidth=2.5)
    axes[0].set_title("(a) Success Rate Convergence", fontweight='bold')
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("SR (%)")
    axes[0].set_ylim(0, 60)

    # (b) Response Length (Inference Density)
    for i, (fname, label) in enumerate(configs.items()):
        df = load_log(fname, label)
        if df is not None:
            axes[1].plot(df['step'], df['len_smooth'], label=label, color=colors[i], linewidth=2.5)
    axes[1].set_title("(b) Reasoning Length Evolution", fontweight='bold')
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Mean Tokens")
    axes[1].set_ylim(100, 350)

    # (c) Exploration Efficiency Index (eta)
    for i, (fname, label) in enumerate(configs.items()):
        df = load_log(fname, label)
        if df is not None:
            axes[2].plot(df['step'], df['eta_smooth'], label=label, color=colors[i], linewidth=2.5)
    axes[2].set_title("(c) Exploration Efficiency ($\eta$)", fontweight='bold')
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Efficiency Index")
    axes[2].set_ylim(0, 3)

    # Legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(PLOT_DIR, "MasterFig2_Ablation_Analysis.svg"))
    plt.savefig(os.path.join(PLOT_DIR, "MasterFig2_Ablation_Analysis.png"), dpi=500)
    print("Generated Master Fig 2 (Ablation/Efficiency)")

if __name__ == "__main__":
    generate_master_fig1_scaling()
    generate_master_fig2_ablation()
