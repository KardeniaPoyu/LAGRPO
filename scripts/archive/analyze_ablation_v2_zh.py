import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_DIR = 'logs'
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────
try:
    b0 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B0_G8_new_metrics.csv')).head(101)
    b1 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B1_G8_new_metrics.csv')).head(101)
    b2 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B2_G8_new_metrics.csv')).head(101)
    b3_path = os.path.join(LOG_DIR, 'grpo_ablation_B3_G8_new_metrics .csv')
    if not os.path.exists(b3_path):
        b3_path = os.path.join(LOG_DIR, 'grpo_ablation_B3_G8_new_metrics.csv') 
    b3 = pd.read_csv(b3_path).head(101)
    b4 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B4_G8_new_metrics.csv')).head(101)
except Exception as e:
    print(f"Error loading CSV files: {e}")
    # Fallback to older files if new ones don't exist
    files = {
        'B0 Vanilla GRPO': 'grpo_ablation_B0_G8_metrics.csv',
        'B1 +Length Penalty': 'grpo_ablation_B1_G8_metrics.csv',
        'B2 +Reward Annealing': 'grpo_ablation_B2_G8_metrics.csv',
        'B3 +Adv Clipping': 'grpo_ablation_B3_G8_metrics.csv',
        'B4 Full LAGRPO (Ours)': 'grpo_ablation_B4_FINAL_G8_metrics.csv'
    }
    dfs = {}
    for name, f in files.items():
        path = os.path.join(LOG_DIR, f)
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path).head(101)
    if not dfs:
        raise e
else:
    dfs = {
        'B0 原生 GRPO':       b0,
        'B1 +长度惩罚':    b1,
        'B2 +奖励退火':  b2,
        'B3 +优势剪切':      b3,
        'B4 LAGRPO 全功能 (本文)': b4,
    }

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    'B0 原生 GRPO':       '#7f7f7f',
    'B1 +长度惩罚':    '#E07B39',
    'B2 +奖励退火':  '#5B8DB8',
    'B3 +优势剪切':      '#6BAF6B',
    'B4 LAGRPO 全功能 (本文)': '#C94040',
}
# Fallback palette keys
for k in list(dfs.keys()):
    if k not in PALETTE:
        if 'B0' in k: PALETTE[k] = '#7f7f7f'
        elif 'B1' in k: PALETTE[k] = '#E07B39'
        elif 'B2' in k: PALETTE[k] = '#5B8DB8'
        elif 'B3' in k: PALETTE[k] = '#6BAF6B'
        elif 'B4' in k: PALETTE[k] = '#C94040'

LW   = {k: 1.5 for k in dfs}
for k in dfs:
    if 'B4' in k: LW[k] = 2.4
LS   = {k: '-' for k in dfs}
for k in dfs:
    if 'B0' in k: LS[k] = '--'
    elif 'B1' in k: LS[k] = ':'
    elif 'B2' in k: LS[k] = '-.'
    elif 'B3' in k: LS[k] = (0,(3,1,1,1))
    elif 'B4' in k: LS[k] = '-'

ZO   = {k: 2 for k in dfs}
for k in dfs:
    if 'B4' in k: ZO[k] = 5

def ema_smooth(series, alpha=0.12):
    return series.ewm(alpha=alpha, adjust=False).mean()

# ── Figure 1: Main 1×3 ────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

fig1, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={'wspace': 0.38})
fig1.patch.set_facecolor('white')

# (a) EMA Success Rate
ax = axes[0]
for name, df in dfs.items():
    steps = df['step'].values
    col = 'ema_success_rate' if 'ema_success_rate' in df.columns else 'success_rate'
    vals  = df[col].values * 100
    ax.plot(steps, vals, color=PALETTE[name], lw=LW[name], ls=LS[name], zorder=ZO[name], alpha=0.93)

ax.set_xlabel('更新步数', fontsize=10)
ax.set_ylabel('EMA 成功率 (%)', fontsize=10)
ax.set_title('(a) EMA 成功率', fontsize=11, fontweight='bold', pad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.1f}%'))
ax.tick_params(labelsize=9)
handles = [mpatches.Patch(color=PALETTE[n], label=n) for n in dfs]
ax.legend(handles=handles, fontsize=7.8, framealpha=0.9, loc='upper left', handlelength=2.2)

# (b) Output Length — Boxplot
ax = axes[1]
box_data  = [df['mean_response_length'].values for df in dfs.values()]
box_names = list(dfs.keys())
bp = ax.boxplot(box_data, patch_artist=True, notch=False, widths=0.52,
                medianprops=dict(color='black', lw=2.2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4, lw=0))

for patch, name in zip(bp['boxes'], box_names):
    patch.set_facecolor(PALETTE[name]); patch.set_alpha(0.80)
for flier, name in zip(bp['fliers'], box_names):
    flier.set(markerfacecolor=PALETTE[name], markeredgecolor=PALETTE[name])

short = ['B0\n原生', 'B1\n+长度', 'B2\n+退火', 'B3\n+剪切', 'B4\nLAGRPO']
ax.set_xticks(range(1, 6)); ax.set_xticklabels(short, fontsize=8.5)
ax.set_ylabel('平均回复长度 (tokens)', fontsize=10)
ax.set_title('(b) 输出长度分布', fontsize=11, fontweight='bold', pad=8)
ax.tick_params(axis='y', labelsize=9)

for i, (name, df) in enumerate(dfs.items(), 1):
    med = np.median(df['mean_response_length'])
    ax.text(i, med + 3, f'{med:.0f}', ha='center', va='bottom', fontsize=7.5, color='#333', fontweight='bold')

# (c) Exploration Efficiency η
ax = axes[2]
for name, df in dfs.items():
    steps = df['step'].values
    sr_col = 'ema_success_rate' if 'ema_success_rate' in df.columns else 'success_rate'
    raw_eta = df[sr_col] / df['mean_response_length'] * 1000
    eta = ema_smooth(raw_eta, alpha=0.15).values
    ax.plot(steps, eta, color=PALETTE[name], lw=LW[name], ls=LS[name], zorder=ZO[name], alpha=0.93)

ax.set_xlabel('更新步数', fontsize=10)
ax.set_ylabel(r'$\eta$ 指数', fontsize=10)
ax.set_title(r'(c) 探索效率 ($\eta$)', fontsize=11, fontweight='bold', pad=8)
ax.tick_params(labelsize=9)

for ax_ in axes:
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    ax_.grid(True, linestyle='--', linewidth=0.45, alpha=0.55)

fig1.savefig(os.path.join(PLOT_DIR, 'ablation_main_v2_zh.png'), dpi=200, bbox_inches='tight', facecolor='white')
print("Main figure saved.")

# ── Figure 2: Appendix — Training Dynamics 1×3 ───────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={'wspace': 0.40})
fig2.patch.set_facecolor('white')
ALPHA_SM = 0.12

# (a) Policy Entropy
ax = axes2[0]
for name, df in dfs.items():
    if 'policy_entropy' not in df.columns: continue
    steps = df['step'].values
    raw   = df['policy_entropy'].values
    smth  = ema_smooth(pd.Series(raw), ALPHA_SM).values
    ax.plot(steps, raw, color=PALETTE[name], lw=0.5, alpha=0.15, zorder=1)
    ax.plot(steps, smth, color=PALETTE[name], lw=LW[name], ls=LS[name], zorder=ZO[name], alpha=0.93)

ax.set_xlabel('更新步数', fontsize=10)
ax.set_ylabel('策略熵 (nats)', fontsize=10)
ax.set_title('(a) 策略熵', fontsize=11, fontweight='bold', pad=8)
ax.tick_params(labelsize=9)
handles = [mpatches.Patch(color=PALETTE[n], label=n) for n in dfs if 'policy_entropy' in dfs[n].columns]
ax.legend(handles=handles, fontsize=7.5, framealpha=0.9, loc='lower right', handlelength=2.2)

# (b) KL Divergence
ax = axes2[1]
for name, df in dfs.items():
    if 'kl_div' not in df.columns: continue
    steps = df['step'].values
    raw   = df['kl_div'].values
    smth  = ema_smooth(pd.Series(raw), ALPHA_SM).values
    ax.plot(steps, raw, color=PALETTE[name], lw=0.5, alpha=0.15, zorder=1)
    ax.plot(steps, smth, color=PALETTE[name], lw=LW[name], ls=LS[name], zorder=ZO[name], alpha=0.93)

ax.axhline(y=10, color='#555', lw=1.0, ls=':', alpha=0.7)
ax.text(98, 10.3, 'KL 阈值', ha='right', va='bottom', fontsize=7, color='#555')
ax.set_xlabel('更新步数', fontsize=10)
ax.set_ylabel('KL 散度', fontsize=10)
ax.set_title('(b) 策略漂移 (KL 散度)', fontsize=11, fontweight='bold', pad=8)
ax.tick_params(labelsize=9)

# (c) Gradient Second Moment — log scale
ax = axes2[2]
for name, df in dfs.items():
    if 'grad_second_moment' not in df.columns: continue
    steps = df['step'].values
    gsm_raw = df['grad_second_moment'].copy()
    raw   = gsm_raw.values
    smth  = ema_smooth(pd.Series(raw), ALPHA_SM).values
    ax.plot(steps, raw, color=PALETTE[name], lw=0.5, alpha=0.15, zorder=1)
    ax.plot(steps, smth, color=PALETTE[name], lw=LW[name], ls=LS[name], zorder=ZO[name], alpha=0.93)

ax.set_yscale('log')
ax.set_xlabel('更新步数', fontsize=10)
ax.set_ylabel('梯度二阶矩 (对数刻度)', fontsize=10)
ax.set_title('(c) 优化方差\n(梯度二阶矩)', fontsize=11, fontweight='bold', pad=8)
ax.tick_params(labelsize=9)

for ax_ in axes2:
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    ax_.grid(True, linestyle='--', linewidth=0.45, alpha=0.55)

fig2.savefig(os.path.join(PLOT_DIR, 'ablation_appendix_v2_zh.png'), dpi=200, bbox_inches='tight', facecolor='white')
print("Appendix figure saved.")

# ── Figure 3: Hallucination Rate comparison ───────────────────────────────────
if all('hallucination_rate' in df.columns for df in dfs.values()):
    fig3, ax3 = plt.subplots(figsize=(7, 4), facecolor='white')
    hal_means = []
    hal_stds  = []
    labels_short = ['B0\n原生\nGRPO', 'B1\n+长度\n惩罚', 'B2\n+奖励\n退火', 'B3\n+优势\n剪切', 'B4\n全功能\nLAGRPO']
    colors = list(PALETTE.values())
    for name, df in dfs.items():
        hal_means.append(df['hallucination_rate'].mean() * 100)
        hal_stds.append(df['hallucination_rate'].std() * 100)
    x = np.arange(len(dfs))
    bars = ax3.bar(x, hal_means, yerr=hal_stds, capsize=4, color=colors, alpha=0.82, width=0.6, error_kw=dict(lw=1.2, ecolor='#333'))
    ax3.set_xticks(x); ax3.set_xticklabels(labels_short, fontsize=9)
    ax3.set_ylabel('平均幻觉率 (%)', fontsize=10)
    ax3.set_title('(d) 各消融配置下的幻觉率对比', fontsize=11, fontweight='bold', pad=8)
    ax3.tick_params(axis='y', labelsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', linestyle='--', linewidth=0.45, alpha=0.55)
    for i, (m, s) in enumerate(zip(hal_means, hal_stds)):
        ax3.text(i, m + s + 0.8, f'{m:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')
    fig3.tight_layout()
    fig3.savefig(os.path.join(PLOT_DIR, 'ablation_hallucination_v2_zh.png'), dpi=200, bbox_inches='tight', facecolor='white')
    print("Hallucination figure saved.")

print("\n=== 所有中文消融分析图表已保存至 plots/ 目录 ===")
