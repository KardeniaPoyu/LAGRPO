import pandas as pd
import numpy as np

def generate_master_ablation_table():
    FILES = {
        'B0: Vanilla':         'logs/grpo_ablation_B0_G8_metrics.csv',
        'B1: +Len Penalty':    'logs/grpo_ablation_B1_G8_metrics.csv',
        'B2: +Annealing':      'logs/grpo_ablation_B2_G8_metrics.csv',
        'B3: +Adv Clipping':   'logs/grpo_ablation_B3_G8_metrics.csv',
        'B4: Full LAGRPO':     'logs/grpo_ablation_B4_FINAL_G8_metrics.csv',
    }
    
    results = []
    
    # 提取基准值用于计算提升
    b0_df = pd.read_csv(FILES['B0: Vanilla'])
    b0_auc_eta = ((b0_df['ema_success_rate'] / b0_df['mean_response_length']) * 1000).sum()

    for label, path in FILES.items():
        df = pd.read_csv(path)
        
        # 1. 成功率指标
        avg_sr = df['ema_success_rate'].mean() * 100
        peak_sr = df['ema_success_rate'].max() * 100
        
        # 2. 长度与稳定性
        avg_len = df['mean_response_length'].mean()
        snr = df['ema_success_rate'].mean() / (df['ema_success_rate'].std() + 1e-9)
        
        # 3. 效率指标
        eta = (df['ema_success_rate'] / df['mean_response_length']) * 1000
        avg_eta = eta.mean()
        auc_eta = eta.sum()
        
        # 4. 提升计算 (Relative to B0)
        eta_gain = (auc_eta / b0_auc_eta - 1) * 100
        
        results.append({
            'Group': label,
            'Avg SR (%)': avg_sr,
            'Peak SR (%)': peak_sr,
            'Avg Length': avg_len,
            'Avg Efficiency (η)': avg_eta,
            'SNR (Stability)': snr,
            'η-AUC Gain (%)': eta_gain
        })
    
    master_df = pd.DataFrame(results)
    print("=== Master Ablation Study Summary Table ===")
    print(master_df.round(3).to_string(index=False))

if __name__ == "__main__":
    generate_master_ablation_table()
