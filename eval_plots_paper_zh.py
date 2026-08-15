"""
eval_plots_paper_zh.py — 论文作图脚本 (中文版)
支持 PPO vs GRPO 对比 + GRPO G 消融实验可视化
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 提升学术图表规范的全局设置 (支持中文)
# ==========================================
# 1.1 矢量图优化设置
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

def set_zh_font():
    # 设置中文字体 (在 style/theme 设置之后调用)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

# ==========================================
# 2. 颜色与线型规范
# ==========================================
ACADEMIC_COLORS = {
    "LAGRPO": "#C94040",
    "B4": "#C94040",
    "PPO": "#2C3E50",
    "B0": "#7F7F7F",
    "BASELINE": "#7F7F7F", 
    "G=4": "#5B8DB8",
    "B2": "#5B8DB8",       
    "G=8": "#6BAF6B",
    "B3": "#6BAF6B",       
    "G=16": "#E07B39",
    "B1": "#E07B39",       
}

LINE_STYLES = {
    "PPO": "--",
    "B0": ":",            
    "DEFAULT": "-"
}

def get_style_for_label(label, all_labels=None):
    target_color = "#95A5A6"
    target_ls = LINE_STYLES["DEFAULT"]
    label_upper = label.upper()
    for key, color in ACADEMIC_COLORS.items():
        if key in label_upper:
            target_color = color
            break
    for key, ls in LINE_STYLES.items():
        if key in label_upper:
            target_ls = ls
            break
    return target_color, target_ls

def load_and_process(filepath, smooth_alpha=0.2, max_steps=None):
    df = pd.read_csv(filepath)
    if 'step' in df.columns:
        df = df.drop_duplicates(subset=['step'], keep='last')
        df = df.sort_values('step').reset_index(drop=True)
        if max_steps is not None:
            df = df[df['step'] <= max_steps].reset_index(drop=True)
    if 'kl_ref' in df.columns and 'kl_div' not in df.columns:
        df = df.rename(columns={'kl_ref': 'kl_div'})
    if 'success_rate' in df.columns and 'mean_response_length' in df.columns:
        df['eta'] = (df['success_rate'] / (df['mean_response_length'] + 1e-5)) * 1000
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'step':
            df[f'{col}_smooth'] = df[col].ewm(alpha=smooth_alpha, adjust=False).mean()
    return df

def create_comparison_plot(ax, dfs, col, title, ylabel, use_log=False, xlabel='更新步数'):
    all_labels = list(dfs.keys())
    for label, df in dfs.items():
        if col in df.columns:
            color, linestyle = get_style_for_label(label, all_labels)
            ax.plot(df['step'], df[col], color=color, alpha=0.25, linewidth=1.0)
            smooth_col = f'{col}_smooth' if f'{col}_smooth' in df.columns else col
            lw = 2.8 if ("LAGRPO" in label or "B4" in label) else 1.8
            ax.plot(df['step'], df[smooth_col], label=label, 
                    color=color, linestyle=linestyle, linewidth=lw, alpha=0.93)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if use_log:
        ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.45, alpha=0.55)
    ax.grid(axis='x', visible=False)

def sort_labels(dfs_dict):
    def sort_key(k):
        nums = [int(s) for s in k if s.isdigit()]
        num = nums[0] if nums else float('inf')
        return (0 if "PPO" in k else 1, num)
    return {k: dfs_dict[k] for k in sorted(dfs_dict.keys(), key=sort_key)}

def plot_ppo_vs_grpo(log_dir='logs', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    dfs = {}
    ppo_file_new = os.path.join(log_dir, 'ppo_metrics_new.csv')
    ppo_file_old = os.path.join(log_dir, 'ppo_metrics.csv')
    if os.path.exists(ppo_file_new):
        dfs['PPO (Critic)'] = load_and_process(ppo_file_new, max_steps=90)
    elif os.path.exists(ppo_file_old):
        dfs['PPO (Critic)'] = load_and_process(ppo_file_old)
    grpo_g8_file = os.path.join(log_dir, 'grpo_G8_metrics.csv')
    grpo_g16_file = os.path.join(log_dir, 'grpo_G16_metrics.csv')
    if os.path.exists(grpo_g8_file):
        dfs['GRPO (G=8)'] = load_and_process(grpo_g8_file, max_steps=90)
    elif os.path.exists(grpo_g16_file):
        dfs['GRPO (G=16)'] = load_and_process(grpo_g16_file, max_steps=90)
    else:
        for f in glob.glob(os.path.join(log_dir, 'grpo_G*_metrics.csv')):
            basename = os.path.basename(f)
            label = basename.replace('_metrics.csv', '').replace('grpo_', '')
            dfs[f'GRPO ({label})'] = load_and_process(f, max_steps=90)
            break
    lagrpo_file = os.path.join(log_dir, 'grpo_B4_G8_metrics.csv')
    if os.path.exists(lagrpo_file):
        dfs['LAGRPO (G=8)'] = load_and_process(lagrpo_file, max_steps=90)
    if not dfs:
        return
    dfs = sort_labels(dfs)
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('PPO 与 GRPO 的扩展性与效率分析 (Arithmetic-24)', fontweight='bold', y=0.98, fontsize=20)

    create_comparison_plot(axes[0, 0], dfs, 'success_rate', '(a) 样本效率（成功率）', '成功率 (%)')
    create_comparison_plot(axes[0, 1], dfs, 'mean_response_length', '(b) 推理长度成本', '平均 Token 数')
    create_comparison_plot(axes[0, 2], dfs, 'eta', '(c) 效率指数 ($\eta$)', '$\eta$ 指数')
    create_comparison_plot(axes[1, 0], dfs, 'kl_div', '(d) 策略漂移（KL 散度）', 'KL 散度')
    create_comparison_plot(axes[1, 1], dfs, 'grad_norm', '(e) 梯度稳定性（L2 范数）', 'L2 范数', use_log=True)
    create_comparison_plot(axes[1, 2], dfs, 'grad_second_moment', '(f) 梯度方差', '二阶矩', use_log=True)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93),
               ncol=len(labels), frameon=True, shadow=True, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(output_dir, 'ppo_vs_grpo_2x3_master_zh.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_g_ablation(log_dir='logs', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    dfs = {}
    for f in glob.glob(os.path.join(log_dir, 'grpo_G*_metrics.csv')):
        basename = os.path.basename(f)
        label = basename.replace('_metrics.csv', '').replace('grpo_', '')
        dfs[f"G={label[1:] if label.startswith('G') else label}"] = load_and_process(f)
    lagrpo_file = os.path.join(log_dir, 'grpo_B4_G8_metrics.csv')
    if os.path.exists(lagrpo_file):
        dfs["LAGRPO (G=8)"] = load_and_process(lagrpo_file)
    if not dfs:
        return
    dfs = sort_labels(dfs)
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('GRPO 组大小 (G) 消融实验：扩展行为分析 (EBS=16/64)', fontweight='bold', y=0.98, fontsize=20)
    create_comparison_plot(axes[0, 0], dfs, 'success_rate', '(a) 样本效率（成功率）', '成功率 (%)')
    create_comparison_plot(axes[0, 1], dfs, 'mean_response_length', '(b) 推理 Token 数 vs. G', '平均 Token 数')
    create_comparison_plot(axes[0, 2], dfs, 'eta', '(c) 组探索效率 ($\eta$)', '$\eta$ 指数')
    create_comparison_plot(axes[1, 0], dfs, 'adv_std', '(d) 优势估计方差', '奖励标准差')
    create_comparison_plot(axes[1, 1], dfs, 'kl_div', '(e) 策略约束（KL 散度）', 'KL 散度')
    create_comparison_plot(axes[1, 2], dfs, 'grad_norm', '(f) 梯度稳定性（L2 范数）', 'L2 范数', use_log=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93),
               ncol=len(labels), frameon=True, shadow=True, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(output_dir, 'grpo_g_ablation_2x3_master_zh.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_lagrpo_ablation(log_dir='logs', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    dfs = {}
    files = {
        "基准 GRPO (B0)": "grpo_ablation_B0_G8_metrics.csv",
        "+长度感知 (B1)": "grpo_ablation_B1_G8_metrics.csv",
        "+退火机制 (B2)": "grpo_ablation_B2_G8_metrics.csv",
        "+优势剪切 (B3)": "grpo_ablation_B3_G8_metrics.csv",
        "LAGRPO 全功能 (B4)": "grpo_ablation_B4_FINAL_G8_metrics.csv"
    }
    for label, filename in files.items():
        f = os.path.join(log_dir, filename)
        if os.path.exists(f):
            dfs[label] = load_and_process(f)
    if not dfs:
        return
    dfs = sort_labels(dfs)
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('LAGRPO 机制消融实验：性能、成本与稳定性 (EBS=64)', fontweight='bold', y=0.98, fontsize=20)
    create_comparison_plot(axes[0, 0], dfs, 'success_rate', '(a) 成功率收敛', '成功率 (%)')
    create_comparison_plot(axes[0, 1], dfs, 'mean_response_length', '(b) 推理轨迹长度', '平均 Token 数')
    create_comparison_plot(axes[0, 2], dfs, 'eta', '(c) 探索效率 ($\eta$)', '$\eta$ 指数')
    create_comparison_plot(axes[1, 0], dfs, 'mean_advantage', '(d) 优势分布偏差', '平均优势')
    create_comparison_plot(axes[1, 1], dfs, 'grad_norm', '(e) 梯度稳定性（L2 范数）', 'L2 范数', use_log=True)
    create_comparison_plot(axes[1, 2], dfs, 'grad_second_moment', '(f) 梯度方差', '二阶矩', use_log=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93),
               ncol=len(labels), frameon=True, shadow=True, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(output_dir, 'lagrpo_ablation_2x3_master_zh.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_eval_summary(log_dir='logs', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    eval_files = glob.glob(os.path.join(log_dir, 'eval_*.jsonl'))
    if not eval_files:
        return
    records = []
    for f in eval_files:
        basename = os.path.basename(f)
        name_parts = basename.replace('eval_', '').replace('.jsonl', '').rsplit('_', 1)
        if len(name_parts) == 2:
            model_name, n_key = name_parts
        else:
            model_name = name_parts[0]
            n_key = "all"
        correct = 0
        total = 0
        with open(f, 'r', encoding='utf-8') as jf:
            for line in jf:
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get('correct', False):
                        correct += 1
                except:
                    pass
        if total > 0:
            records.append({'model': model_name, 'n_key': n_key, 'accuracy': correct / total})
    if not records:
        return
    df_records = pd.DataFrame(records)
    df_records['指标'] = df_records['n_key'].apply(lambda x: f"N={x}" if str(x).isdigit() else str(x).upper())
    df_records['准确率'] = df_records['accuracy'] * 100
    all_models = df_records['model'].unique().tolist()
    all_models.sort(key=lambda x: (0 if 'ppo' in x.lower() else 1, x))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_records, x='指标', y='准确率', hue='model', hue_order=all_models, ax=ax, palette='colorblind', edgecolor='white')
    ax.set_title('测试集上的零样本评估成功率', fontweight='bold', pad=20)
    ax.set_ylabel('成功率 (%)')
    ax.set_xlabel('难度等级')
    max_acc = df_records['准确率'].max()
    ax.set_ylim(0, max(max_acc * 1.2, 5.0))
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9, color='#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='#E0E0E0')
    ax.grid(axis='x', visible=False)
    ax.legend(title='', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_summary_zh.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_difficulty_curve(log_dir='logs', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    eval_files = glob.glob(os.path.join(log_dir, 'eval_*.jsonl'))
    if not eval_files:
        return
    records = []
    for f in eval_files:
        basename = os.path.basename(f)
        raw_name = basename.replace('eval_', '').replace('.jsonl', '')
        name_parts = raw_name.rsplit('_', 1)
        if len(name_parts) == 2 and (name_parts[1].isdigit() or name_parts[1] == 'all'):
            model_full_name, n_key = name_parts
        else:
            model_full_name = raw_name
            n_key = "all"
        if n_key == "all":
            continue
        n_val = int(n_key)
        base_model = model_full_name
        seed = 1
        if "_seed_" in model_full_name:
            parts = model_full_name.split("_seed_")
            base_model = parts[0]
            try:
                seed = int(parts[1])
            except:
                pass
        display_name = base_model.upper().replace('_FINAL', '')
        correct = 0
        total = 0
        with open(f, 'r', encoding='utf-8') as jf:
            for line in jf:
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get('correct', False):
                        correct += 1
                except:
                    pass
        if total > 0:
            records.append({
                '模型': display_name, 
                '难度 (N)': n_val, 
                'Seed': seed,
                '成功率 (%)': (correct / total) * 100
            })
    if not records:
        return
    df = pd.DataFrame(records)
    df.sort_values('难度 (N)', inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x='难度 (N)', y='成功率 (%)', hue='模型', style='模型', markers=True, dashes=False, errorbar='sd', linewidth=2.5, markersize=10, palette='colorblind', ax=ax)
    ax.set_title('零样本成功率随难度的变化 (均值 ± 1 标准差)', fontweight='bold', pad=20)
    ax.set_yscale('symlog', linthresh=1.0)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_yticks([0, 1, 5, 10, 20, 50, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='', frameon=True, edgecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diff_success_curve_zh.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_heatmap(log_dir='logs', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    eval_files = glob.glob(os.path.join(log_dir, 'eval_*.jsonl'))
    if not eval_files:
        return
    records = []
    for f in eval_files:
        basename = os.path.basename(f)
        raw_name = basename.replace('eval_', '').replace('.jsonl', '')
        name_parts = raw_name.rsplit('_', 1)
        if len(name_parts) == 2 and (name_parts[1].isdigit() or name_parts[1] == 'all'):
            model_full_name, n_key = name_parts
        else:
            model_full_name = raw_name
            n_key = "all"
        if n_key == "all":
            continue
        n_val = int(n_key)
        base_model = model_full_name
        if "_seed_" in model_full_name:
            base_model = model_full_name.split("_seed_")[0]
        display_name = base_model.upper().replace('_FINAL', '')
        correct = 0
        total = 0
        with open(f, 'r', encoding='utf-8') as jf:
            for line in jf:
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get('correct', False):
                        correct += 1
                except:
                    pass
        if total > 0:
            records.append({
                '模型': display_name, 
                '难度 (N)': f"N={n_val}", 
                '成功率': (correct / total) * 100
            })
    if not records:
        return
    df = pd.DataFrame(records)
    agg_df = df.groupby(['模型', '难度 (N)'])['成功率'].mean().reset_index()
    pivot_df = agg_df.pivot(index='模型', columns='难度 (N)', values='成功率')
    models = list(pivot_df.index)
    models.sort(key=lambda x: (0 if 'PPO' in x else 1, x))
    pivot_df = pivot_df.reindex(models)
    fig, ax = plt.subplots(figsize=(8, max(4, len(models)*0.8)))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '成功率 (%)'}, linewidths=1, linecolor='white', ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_title("跨难度等级的零样本成功率", pad=30, fontweight='bold')
    ax.set_ylabel("模型架构")
    ax.set_xlabel("任务难度")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_heatmap_zh.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="生成论文图表 (中文版)")
    parser.add_argument("--ablation", action="store_true", help="生成 G 消融图")
    parser.add_argument("--lagrpo", action="store_true", help="生成 LAGRPO 消融图")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()

    if args.all or (not args.ablation and not args.lagrpo):
        set_zh_font()
        plot_ppo_vs_grpo(args.log_dir, args.output_dir)
    if args.all or args.ablation:
        set_zh_font()
        plot_g_ablation(args.log_dir, args.output_dir)
    if args.all or args.lagrpo:
        set_zh_font()
        plot_lagrpo_ablation(args.log_dir, args.output_dir)
    if args.all:
        set_zh_font()
        plot_eval_summary(args.log_dir, args.output_dir)
        plot_difficulty_curve(args.log_dir, args.output_dir)
        plot_success_heatmap(args.log_dir, args.output_dir)
    print(f"\n所有图表已生成至 '{args.output_dir}/' 目录")

if __name__ == "__main__":
    main()
