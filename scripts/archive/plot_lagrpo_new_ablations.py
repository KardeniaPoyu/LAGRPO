import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

def load_and_process(file_path, max_steps=100):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    try:
        df = pd.read_csv(file_path)
        # 裁剪到对齐的步数，方便对比
        df = df[df['step'] <= max_steps].copy()
        
        # 计算 EMA 成功率
        df['ema_sr'] = df['success_rate'].ewm(alpha=0.1).mean()
        
        # 计算效率指数 eta = SR / Length * 1000
        # 避免除以 0
        df['efficiency'] = (df['success_rate'] / df['mean_response_length'].replace(0, np.nan)) * 1000
        df['ema_efficiency'] = df['efficiency'].ewm(alpha=0.1).mean()
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    log_dir = 'logs'
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件映射
    file_map = {
        "Baseline (B0)": "grpo_ablation_B0_G8_new_metrics.csv",
        "+Length (B1)": "grpo_ablation_B1_G8_new_metrics.csv",
        "+Anneal (B2)": "grpo_ablation_B2_G8_new_metrics.csv",
        "+Clip (B3)": "grpo_ablation_B3_G8_new_metrics .csv", # 注意这个空格
        "Full LAGRPO (B4)": "grpo_ablation_B4_G8_new_metrics.csv",
        "PPO (Ref)": "ppo_metrics_new.csv"
    }
    
    dfs = {}
    for label, filename in file_map.items():
        path = os.path.join(log_dir, filename)
        df = load_and_process(path)
        if df is not None:
            dfs[label] = df
            
    if not dfs:
        print("No data found to plot.")
        return

    # 创建 2x3 画布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle('LAGRPO Iron Triangle Ablation Analysis (N=4 task)', fontweight='bold')
    
    colors = {
        "Baseline (B0)": "#7f8c8d",      # 灰色
        "+Length (B1)": "#3498db",       # 蓝色
        "+Anneal (B2)": "#9b59b6",       # 紫色
        "+Clip (B3)": "#2ecc71",         # 绿色
        "Full LAGRPO (B4)": "#e67e22",   # 橙色 (高亮)
        "PPO (Ref)": "#e74c3c"           # 红色
    }
    
    # 1. Success Rate (EMA)
    ax = axes[0, 0]
    for label, df in dfs.items():
        ax.plot(df['step'], df['ema_sr'], label=label, color=colors.get(label, None), linewidth=2 if "B4" in label else 1.5)
    ax.set_title('(a) Success Rate (EMA)')
    ax.set_ylabel('SR')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Response Length
    ax = axes[0, 1]
    for label, df in dfs.items():
        # 同样使用 EMA 平滑长度
        ax.plot(df['step'], df['mean_response_length'].ewm(alpha=0.1).mean(), label=label, color=colors.get(label, None))
    ax.set_title('(b) Mean Response Length')
    ax.set_ylabel('Tokens')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 3. Efficiency Index (eta)
    ax = axes[0, 2]
    for label, df in dfs.items():
        ax.plot(df['step'], df['ema_efficiency'], label=label, color=colors.get(label, None))
    ax.set_title('(c) Efficiency Index (η)')
    ax.set_ylabel('SR per 1k Tokens')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 4. KL Divergence
    ax = axes[1, 0]
    for label, df in dfs.items():
        if 'kl_div' in df.columns:
            ax.plot(df['step'], df['kl_div'].ewm(alpha=0.1).mean(), label=label, color=colors.get(label, None))
    ax.set_title('(d) KL Divergence (Stability)')
    ax.set_ylabel('KL')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 5. Advantage STD
    ax = axes[1, 1]
    for label, df in dfs.items():
        if 'adv_std' in df.columns:
            ax.plot(df['step'], df['adv_std'].ewm(alpha=0.1).mean(), label=label, color=colors.get(label, None))
    ax.set_title('(e) Advantage STD (Exploration Variance)')
    ax.set_ylabel('STD')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 6. Policy Entropy
    ax = axes[1, 2]
    for label, df in dfs.items():
        if 'policy_entropy' in df.columns:
            ax.plot(df['step'], df['policy_entropy'].ewm(alpha=0.1).mean(), label=label, color=colors.get(label, None))
    ax.set_title('(f) Policy Entropy')
    ax.set_ylabel('Entropy')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 图例设置在下方
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(dfs), bbox_to_anchor=(0.5, -0.05))
    
    output_path = os.path.join(output_dir, 'lagrpo_new_ablation_2x3.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
