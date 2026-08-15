
import pandas as pd
import numpy as np
import os

def get_stats(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    # Get last 20 steps for convergence analysis
    last_20 = df.tail(20)
    
    # Simple stats
    stats = {
        'final_success_mean': last_20['success_rate'].mean(),
        'final_success_std': last_20['success_rate'].std(),
        'kl_mean': df['kl_div' if 'kl_div' in df.columns else 'kl_ref'].mean(),
        'kl_std': df['kl_div' if 'kl_div' in df.columns else 'kl_ref'].std(),
        'kl_var': df['kl_div' if 'kl_div' in df.columns else 'kl_ref'].var(),
        'avg_adv_std': df['adv_std'].mean() if 'adv_std' in df.columns else 0,
        'avg_len': df['mean_response_length'].mean()
    }
    
    # Correlation between length and success
    if 'success_rate' in df.columns and 'mean_response_length' in df.columns:
        corr = df['success_rate'].corr(df['mean_response_length'])
        stats['len_success_corr'] = corr
        
    return stats

log_dir = 'd:/Personal/Documents/GitHub/SLM-RL-Comparation/logs'
files = {
    'ppo': 'ppo_metrics.csv',
    'g4': 'grpo_G4_metrics.csv',
    'g8': 'grpo_G8_metrics.csv',
    'g16': 'grpo_G16_metrics.csv'
}

all_results = {}
for key, filename in files.items():
    res = get_stats(os.path.join(log_dir, filename))
    if res:
        all_results[key] = res

import json
print(json.dumps(all_results, indent=2))
