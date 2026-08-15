
import pandas as pd
import numpy as np
import os
import json

def get_stats(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        last_20 = df.tail(20)
        
        kl_col = 'kl_div' if 'kl_div' in df.columns else 'kl_ref'
        
        stats = {
            'final_success_mean': float(last_20['success_rate'].mean()),
            'final_success_std': float(last_20['success_rate'].std()),
            'kl_mean': float(df[kl_col].mean()),
            'kl_std': float(df[kl_col].std()),
            'kl_var': float(df[kl_col].var()),
            'avg_adv_std': float(df['adv_std'].mean()) if 'adv_std' in df.columns else 0.0,
            'avg_len': float(df['mean_response_length'].mean())
        }
        
        if 'success_rate' in df.columns and 'mean_response_length' in df.columns:
            stats['len_success_corr'] = float(df['success_rate'].corr(df['mean_response_length']))
        return stats
    except Exception as e:
        return {"error": str(e)}

log_dir = 'd:/Personal/Documents/GitHub/SLM-RL-Comparation/logs'
files = {
    'ppo': 'ppo_metrics.csv',
    'g4': 'grpo_G4_metrics.csv',
    'g8': 'grpo_G8_metrics.csv',
    'g16': 'grpo_G16_metrics.csv'
}

results = {}
for key, filename in files.items():
    stats = get_stats(os.path.join(log_dir, filename))
    if stats:
        results[key] = stats

with open('final_stats.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Stats saved to final_stats.json")
