import pandas as pd
import os

LOG_DIR = 'logs'
files = {
    'B0': 'grpo_ablation_B0_G8_new_metrics.csv',
    'B1': 'grpo_ablation_B1_G8_new_metrics.csv',
    'B2': 'grpo_ablation_B2_G8_new_metrics.csv',
    'B3': 'grpo_ablation_B3_G8_new_metrics .csv',
    'B4': 'grpo_ablation_B4_G8_new_metrics.csv'
}

print(f"{'Config':<5} | {'Stat':<10} | {'Stable SR (Tail 50)':<12} | {'Len Med (Full)':<12} | {'Len Mean (Tail 50)':<14} | {'Len Med (Tail 50)':<14}")
print("-" * 85)

for name, fname in files.items():
    df = pd.read_csv(os.path.join(LOG_DIR, fname)).head(101)
    stable = df.tail(50)
    
    sr_stable = stable['ema_success_rate'].mean()
    len_med_full = df['mean_response_length'].median()
    len_mean_stable = stable['mean_response_length'].mean()
    len_med_stable = stable['mean_response_length'].median()
    
    # Calculate different eta candidates
    eta1 = (sr_stable / len_med_full) * 1000
    eta2 = (sr_stable / len_mean_stable) * 1000
    
    print(f"{name:<5} | Values     | {sr_stable:<12.4f} | {len_med_full:<12.1f} | {len_mean_stable:<14.1f} | {len_med_stable:<14.1f}")
    print(f"{name:<5} | Eta (SR/L) | Eta1: {eta1:<8.3f} | Eta2 (Stable): {eta2:<8.3f}")
    print("-" * 85)
