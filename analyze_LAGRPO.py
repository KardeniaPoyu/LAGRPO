import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Data loading ──────────────────────────────────────────────────────────────
configs = {
    'Baseline':       '/mnt/user-data/uploads/grpo_ablation_B0_G8_metrics.csv',
    'Step Anneal':    '/mnt/user-data/uploads/grpo_ablation_B1_G8_metrics.csv',
    'Length Penalty': '/mnt/user-data/uploads/grpo_ablation_B2_G8_metrics.csv',
    'Both':           '/mnt/user-data/uploads/grpo_ablation_B3_G8_metrics.csv',
    'LAGRPO Final':   '/mnt/user-data/uploads/grpo_ablation_B4_FINAL_G8_metrics.csv',
}

dfs = {k: pd.read_csv(v) for k, v in configs.items()}

# ── Palette — academic, colourblind-safe ──────────────────────────────────────
palette = {
    'Baseline':       '#7f7f7f',   # neutral grey
    'Step Anneal':    '#5B8DB8',   # steel blue
    'Length Penalty': '#E07B39',   # burnt orange
    'Both':           '#6BAF6B',   # muted green
    'LAGRPO Final':   '#C94040',   # deep red  ← hero curve
}
lw = {'Baseline': 1.4, 'Step Anneal': 1.4,
      'Length Penalty': 1.4, 'Both': 1.4, 'LAGRPO Final': 2.2}
ls = {'Baseline': '--', 'Step Anneal': ':', 'Length Penalty': '-.',
      'Both': (0,(3,1,1,1)), 'LAGRPO Final': '-'}
zo = {'Baseline': 2, 'Step Anneal': 2, 'Length Penalty': 2,
      'Both': 2, 'LAGRPO Final': 4}

# ── Figure layout ─────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font='DejaVu Sans')
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                         gridspec_kw={'wspace': 0.38})
fig.patch.set_facecolor('white')

# ─────────────────────────────────────────────────────────────────────────────
# (a) EMA Success Rate
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
for name, df in dfs.items():
    ax.plot(df['step'], df['ema_success_rate'] * 100,
            color=palette[name], lw=lw[name], ls=ls[name],
            zorder=zo[name], label=name, alpha=0.92)

ax.set_xlabel('Update Step', fontsize=10)
ax.set_ylabel('EMA Success Rate (%)', fontsize=10)
ax.set_title('(a) EMA Success Rate', fontsize=11, fontweight='bold', pad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
ax.tick_params(labelsize=9)
ax.set_xlim(0, 150)

# legend inside (a)
handles = [mpatches.Patch(color=palette[n], label=n) for n in configs]
ax.legend(handles=handles, fontsize=8, framealpha=0.85,
          loc='upper left', handlelength=2.0)

# ─────────────────────────────────────────────────────────────────────────────
# (b) Output Length Distribution — Boxplot
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]

box_data  = [dfs[n]['mean_response_length'].values for n in configs]
box_names = list(configs.keys())

bp = ax.boxplot(
    box_data,
    patch_artist=True,
    notch=False,
    widths=0.55,
    medianprops=dict(color='black', lw=2.0),
    whiskerprops=dict(lw=1.2),
    capprops=dict(lw=1.2),
    flierprops=dict(marker='o', markersize=3.5, alpha=0.5, lw=0),
)

for patch, name in zip(bp['boxes'], box_names):
    patch.set_facecolor(palette[name])
    patch.set_alpha(0.82)
for flier, name in zip(bp['fliers'], box_names):
    flier.set(markerfacecolor=palette[name], markeredgecolor=palette[name])

# short tick labels to avoid overlap
short = ['Base', 'Anneal', 'Len\nPen', 'Both', 'LAGRPO\nFinal']
ax.set_xticks(range(1, len(box_names) + 1))
ax.set_xticklabels(short, fontsize=8.5)
ax.set_ylabel('Mean Response Length (tokens)', fontsize=10)
ax.set_title('(b) Output Length Distribution', fontsize=11, fontweight='bold', pad=8)
ax.tick_params(axis='y', labelsize=9)

# annotate medians
for i, name in enumerate(box_names, 1):
    med = np.median(dfs[name]['mean_response_length'])
    ax.text(i, med + 3, f'{med:.0f}', ha='center', va='bottom',
            fontsize=7.5, color='#333333', fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# (c) Exploration Efficiency η
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[2]

for name, df in dfs.items():
    eta = (df['ema_success_rate'] / df['mean_response_length']) * 1000
    ax.plot(df['step'], eta,
            color=palette[name], lw=lw[name], ls=ls[name],
            zorder=zo[name], alpha=0.92)

ax.set_xlabel('Update Step', fontsize=10)
ax.set_ylabel(r'$\eta = \frac{\mathrm{Success\,Rate}}{\mathrm{Mean\,Length}} \times 10^3$',
              fontsize=10)
ax.set_title(r'(c) Exploration Efficiency ($\eta$)', fontsize=11,
             fontweight='bold', pad=8)
ax.tick_params(labelsize=9)
ax.set_xlim(0, 150)

# ─────────────────────────────────────────────────────────────────────────────
# Final polish
# ─────────────────────────────────────────────────────────────────────────────
for ax_ in axes:
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    ax_.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

plt.savefig('/mnt/user-data/outputs/ablation_comparison.pdf',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/ablation_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved.")