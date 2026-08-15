import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.signal import savgol_filter

# 设置顶刊学术风格
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('ggplot') # Fallback

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

def process_data(log_dir="logs"):
    files = {
        "B0: Baseline GRPO": "grpo_ablation_B0_G8_metrics.csv",
        "B1: +Length Aware": "grpo_ablation_B1_G8_metrics.csv",
        "B2: +Annealing": "grpo_ablation_B2_G8_metrics.csv",
        "B4: Full LAGRPO": "grpo_ablation_B4_FINAL_G8_metrics.csv"
    }
    
    results = {}
    print("Reading experiment logs...")
    for name, filename in files.items():
        path = os.path.join(log_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure consistency
            df = df.head(150) # Cut to 150 steps for fair comparison
            
            # Normalize step if it isn't starting from 0
            df['step'] = range(len(df))
            
            # scaling
            df['success_rate_pct'] = df['success_rate'] * 100
            # Calculate exploration efficiency eta = (SR / Length) * 1000
            df['eta'] = (df['success_rate'] / (df['mean_response_length'] + 1e-5)) * 1000
            results[name] = df
            print(f" - Loaded {name}: {len(df)} steps")
        else:
            print(f" - Warning: {filename} missing.")
    return results

def plot_journal_figure(results, save_path="plots/Fig4_Ablation_Analysis.png"):
    if not results:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    # Using academic palette
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
        # Variance shade
        ax0.fill_between(x, y_smooth - np.std(y)*0.2, y_smooth + np.std(y)*0.2, color=color, alpha=0.1)
        # Sparse markers
        mark_idx = np.linspace(0, len(x)-1, 10, dtype=int)
        ax0.plot(x[mark_idx], y_smooth[mark_idx], color=color, marker=marker, markersize=9, linestyle='None')

    ax0.set_title("A. Performance Convergence", loc='center', fontweight='bold', pad=15)
    ax0.set_ylabel("Success Rate (%)", fontsize=15)
    ax0.set_xlabel("Training Steps", fontsize=15)
    ax0.set_ylim(0, 50)

    # 2. Token Length Dynamics (Anti-Bloat Proof)
    ax1 = axes[1]
    for (name, df), color in zip(results.items(), colors):
        y = df['mean_response_length'].values
        window = min(15, len(y))
        if window % 2 == 0: window -= 1
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)
        ax1.plot(df['step'], y_smooth, color=color, linewidth=3)
        ax1.fill_between(df['step'], y_smooth, y, color=color, alpha=0.1)

    ax1.set_title("B. Inference Density (Tokens)", loc='center', fontweight='bold', pad=15)
    ax1.set_ylabel("Mean Response Length", fontsize=15)
    ax1.set_xlabel("Training Steps", fontsize=15)
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

    ax2.set_title("C. Exploration Score ($\eta$)", loc='center', fontweight='bold', pad=15)
    ax2.set_ylabel(r"$\eta$ (Efficiency Index)", fontsize=15)
    ax2.set_xlabel("Training Steps", fontsize=15)
    ax2.set_ylim(0, 3)

    # Annotation of LAGRPO superiority
    b4_final_eta = results["B4: Full LAGRPO"]['eta'].tail(10).mean()
    ax2.annotate(f'LAGRPO Efficient Plateau', 
                 xy=(140, b4_final_eta), xytext=(40, 2.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                 fontsize=12, fontweight='bold')

    # Legend across the bottom
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), 
               frameon=True, shadow=True, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.25)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n[REPORT] Final Figures generated in {os.path.dirname(save_path)}/")
    print(f" - High-res PNG: {save_path}")
    print(f" - Vectorized PDF: {save_path.replace('.png', '.pdf')}")

if __name__ == "__main__":
    data = process_data()
    if data:
        plot_journal_figure(data)
        
        # Summary statistics calculation
        print("\n" + "="*40)
        print("   ABLATION STUDY SUMMARY (Final Step)")
        print("="*40)
        for name, df in data.items():
            last_sr = df['success_rate_pct'].tail(10).mean()
            last_len = df['mean_response_length'].tail(10).mean()
            last_eta = df['eta'].tail(10).mean()
            print(f"[{name}]")
            print(f"  - Final Success Rate (Top-10 Avg): {last_sr:.2f}%")
            print(f"  - Final Token Length (Avg): {last_len:.1f}")
            print(f"  - Exploration Efficiency (eta): {last_eta:.3f}")
        print("="*40)
