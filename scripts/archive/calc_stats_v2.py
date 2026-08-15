
import pandas as pd
import numpy as np
import os

def get_stats(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    # Get last 20 steps
    last_20 = df.tail(20)
    
    kl_col = 'kl_div' if 'kl_div' in df.columns else 'kl_ref'
    
    stats = {
        'final_success_mean': last_20['success_rate'].mean(),
        'final_success_std': last_20['success_rate'].std(),
        'kl_mean': df[kl_col].mean(),
        'kl_std': df[kl_col].std(),
        'kl_var': df[kl_col].var(),
        'avg_adv_std': df['adv_std'].mean() if 'adv_std' in df.columns else 0,
        'avg_len': df['mean_response_length'].mean()
    }
    
    if 'success_rate' in df.columns and 'mean_response_length' in df.columns:
        stats['len_success_corr'] = df['success_rate'].corr(df['mean_response_length'])
        
    return stats

log_dir = 'd:/Personal/Documents/GitHub/SLM-RL-Comparation/logs'
files = {
    'ppo': 'ppo_metrics.csv',
    'g4': 'grpo_G4_metrics.csv',
    'g8': 'grpo_G8_metrics.csv',
    'g16': 'grpo_G16_metrics.csv'
}

for key, filename in files.items():
    res = get_stats(os.path.join(log_dir, filename))
    if res:
        print(f"--- {key} ---")
        for k, v in res.items():
            print(f"{k}: {v:.4f}")
