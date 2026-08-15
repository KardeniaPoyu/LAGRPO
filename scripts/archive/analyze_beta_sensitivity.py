import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import re

# 设置绘图风格 (学术版)
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix'
})

def load_final_stats(log_dir, group_size=16):
    """
    搜索指定目录下所有的 beta_*_metrics.csv，提取最后 10 步的平均值作为收敛点。
    """
    pattern = os.path.join(log_dir, f"grpo_beta_*_G{group_size}_metrics.csv")
    files = glob.glob(pattern)
    
    results = []
    
    for f in files:
        # 从文件名提取 beta 值
        match = re.search(r'beta_([\d\.]+)', f)
        if not match:
            continue
        beta = float(match.group(1))
        
        try:
            df = pd.read_csv(f)
            if len(df) < 5:
                continue
            
            # 取收敛阶段 (最后 10% 或最后 10 步)
            tail = df.tail(10)
            avg_sr = tail['success_rate'].mean()
            avg_len = tail['mean_response_length'].mean()
            
            results.append({
                'beta': beta,
                'success_rate': avg_sr,
                'length': avg_len,
                'efficiency': (avg_sr / avg_len * 1000) if avg_len > 0 else 0
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.DataFrame(results).sort_values('beta')

def plot_sensitivity(df, output_path):
    if df.empty:
        print("No data available to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # X 轴: Beta
    x = df['beta']

    # 左轴: Success Rate
    color1 = 'tab:red'
    ax1.set_xlabel(r'Length Penalty Coefficient $\beta$')
    ax1.set_ylabel('Final Success Rate', color=color1)
    line1 = ax1.plot(x, df['success_rate'], 'o-', color=color1, linewidth=2, label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(df['success_rate']) * 1.2)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 右轴: Response Length
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Mean Response Length (Tokens)', color=color2)
    line2 = ax2.plot(x, df['length'], 's--', color=color2, linewidth=2, label='Response Length')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(df['length']) * 1.1)

    # 标注三阶段
    if len(x) >= 3:
        # 情况 1: 退化区 (最左侧)
        ax1.annotate('Regime I: Degeneration\n(Small $\\beta$)', xy=(x.iloc[0], df['success_rate'].iloc[0]), 
                     xytext=(x.iloc[0]+0.05, df['success_rate'].iloc[0]*0.8),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
        
        # 情况 2: 逆势增长点 (通常在中间)
        best_idx = df['efficiency'].idxmax()
        ax1.annotate('Regime II: Optimal Growth\n(High Efficiency)', xy=(df.loc[best_idx, 'beta'], df.loc[best_idx, 'success_rate']), 
                     xytext=(df.loc[best_idx, 'beta'], df.loc[best_idx, 'success_rate']*1.1),
                     ha='center', fontweight='bold', color='darkgreen')
        
        # 情况 3: 强制缩短区
        ax1.annotate('Regime III: Forced Shortening', xy=(x.iloc[-1], df['success_rate'].iloc[-1]), 
                     xytext=(x.iloc[-1]-0.2, df['success_rate'].iloc[-1]+0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.title('Sensitivity Analysis of LAGRPO Length Penalty (Beta)', fontweight='bold', pad=15)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def main():
    log_dir = 'logs/beta_sweep'
    output_dir = 'plots/beta_sensitivity'
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_final_stats(log_dir)
    print("Aggregate Statistics:")
    print(df)
    
    output_file = os.path.join(output_dir, 'beta_three_regimes.png')
    plot_sensitivity(df, output_file)
    
    # 保存数据 CSV 方便后续绘图
    df.to_csv(os.path.join(output_dir, 'beta_summary.csv'), index=False)

if __name__ == "__main__":
    main()
