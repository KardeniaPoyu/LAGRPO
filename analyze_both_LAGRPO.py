import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── Config (Mapped to Mixed Difficulty Experiment 2 logs) ─────────────────────
FILES = {
    'Vanilla GRPO (B0)':    'logs/grpo_ablation_B0_G8_metrics.csv',
    '+Length Penalty (B1)': 'logs/grpo_ablation_B1_G8_metrics.csv',
    '+Annealing (B2)':      'logs/grpo_ablation_B2_G8_metrics.csv',
    '+Adv Clipping (B3)':   'logs/grpo_ablation_B3_G8_metrics.csv',
    'Full LAGRPO (B4)':     'logs/grpo_ablation_B4_FINAL_G8_metrics.csv',
}

PALETTE = {
    'Vanilla GRPO (B0)':    '#7f7f7f',
    '+Length Penalty (B1)': '#E07B39',
    '+Annealing (B2)':      '#5B8DB8',
    '+Adv Clipping (B3)':   '#6BAF6B',
    'Full LAGRPO (B4)':     '#C94040',
}
LW = {k: 1.5 for k in FILES}; LW['Full LAGRPO (B4)'] = 2.4
LS = {
    'Vanilla GRPO (B0)':    '--',
    '+Length Penalty (B1)': ':',
    '+Annealing (B2)':      '-.',
    '+Adv Clipping (B3)':   (0,(3,1,1,1)),
    'Full LAGRPO (B4)':     '-',
}
ZO = {k: 2 for k in FILES}; ZO['Full LAGRPO (B4)'] = 5

os.makedirs('plots', exist_ok=True)
dfs = {k: pd.read_csv(v) for k, v in FILES.items() if os.path.exists(v)}
active_names = [n for n in FILES.keys() if n in dfs]

def ema(series, alpha=0.15):
    return series.ewm(alpha=alpha, adjust=False).mean()

sns.set_theme(style='whitegrid', font='DejaVu Sans')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main (1×3)
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={'wspace': 0.38})
fig1.patch.set_facecolor('white')

# (a) Success Rate
ax = axes[0]
for name in active_names:
    df = dfs[name]
    ax.plot(df['step'], df['ema_success_rate'] * 100, 
            color=PALETTE[name], lw=LW[name], ls=LS[name], 
            zorder=ZO[name], alpha=0.93)
ax.set_ylabel('EMA Success Rate (%)', fontsize=10)
ax.set_title('(a) EMA Success Rate (Mixed Difficulty)', fontsize=11, fontweight='bold', pad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
ax.legend(handles=[mpatches.Patch(color=PALETTE[n], label=n) for n in active_names], 
          fontsize=7.8, loc='upper left', framealpha=0.9)

# (b) Length Distribution
ax = axes[1]
box_data = [dfs[n]['mean_response_length'].values for n in active_names]
bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.55)
for patch, name in zip(bp['boxes'], active_names):
    patch.set_facecolor(PALETTE[name]); patch.set_alpha(0.8)
ax.set_ylabel('Mean Response Length (tokens)', fontsize=10)
ax.set_xticks(range(1, len(active_names)+1))
ax.set_xticklabels(['B0', 'B1', 'B2', 'B3', 'B4'], fontsize=9)
ax.set_title('(b) Output Length Distribution', fontsize=11, fontweight='bold', pad=8)

# (c) Efficiency
ax = axes[2]
for name in active_names:
    df = dfs[name]
    eta = (df['ema_success_rate'] / df['mean_response_length']) * 1000
    ax.plot(df['step'], ema(eta), color=PALETTE[name], lw=LW[name], ls=LS[name], zorder=ZO[name])
ax.set_ylabel(r'$\eta$ Index ($\times 10^3$)', fontsize=10)
ax.set_title('(c) Exploration Efficiency ($\eta$)', fontsize=11, fontweight='bold', pad=8)

for ax_ in axes:
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
    ax_.grid(True, linestyle='--', linewidth=0.5, alpha=0.55)

fig1.savefig('plots/ablation_main.png', dpi=200, bbox_inches='tight')
print("Main figure saved to plots/ablation_main.png")

# ── Summary Table (Table 5-4) ─────────────────────────────────────────────────
print("\n=== Table 5-4: LAGRPO Ablation Study (Mixed Difficulty N=3,4,5,6) ===")
print(f"{'Config':<24} | {'Stable SR':>10} | {'Peak SR':>10} | {'Med Length':>10} | {'Halluc':>8} | {'KL':>8} | {'Eta':>8}")
print("-" * 100)

for name in active_names:
    df = dfs[name]
    s50 = df.tail(50)
    sr_st = s50['ema_success_rate'].mean()
    sr_mx = df['ema_success_rate'].max()
    len_med = df['mean_response_length'].median()
    len_st_med = s50['mean_response_length'].median()
    halluc = df['hallucination_rate'].mean()
    kl = df['kl_div'].mean()
    eta = (sr_st / (len_st_med + 1e-6)) * 1000
    print(f"{name:<24} | {sr_st:>10.4f} | {sr_mx:>10.4f} | {len_med:>10.1f} | {halluc:>8.3f} | {kl:>8.2f} | {eta:>8.3f}")