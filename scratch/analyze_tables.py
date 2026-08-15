import pandas as pd
import numpy as np
import os

log_dir = r"d:\Personal\Documents\GitHub\SLM-RL-Comparation\logs"

files_n4 = {
    "B0": "grpo_ablation_B0_G8_new_metrics.csv",
    "B1": "grpo_ablation_B1_G8_new_metrics.csv",
    "B2": "grpo_ablation_B2_G8_new_metrics.csv",
    "B3": "grpo_ablation_B3_G8_new_metrics .csv",
    "B4": "grpo_ablation_B4_G8_new_metrics.csv"
}

files_mix = {
    "B0": "grpo_ablation_B0_G8_metrics.csv",
    "B1": "grpo_ablation_B1_G8_metrics.csv",
    "B2": "grpo_ablation_B2_G8_metrics.csv",
    "B3": "grpo_ablation_B3_G8_metrics.csv",
    "B4": "grpo_ablation_B4_FINAL_G8_metrics.csv"
}

def analyze_file(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    # Filter out potential repeated headers if any
    df = df[df['step'] != 'step'].apply(pd.to_numeric, errors='ignore')
    
    # Peak EMA
    ema_peak = df['ema_success_rate'].max()
    
    # Success Rate (Stable) - Last 50 steps MEAN
    # The user's image shows "均值" to be the Stable Success.
    ema_stable = df['ema_success_rate'].iloc[-50:].mean() if len(df) > 50 else df['ema_success_rate'].mean()
    
    # Success Rate (Full Mean)
    ema_mean_full = df['ema_success_rate'].mean()

    # Median Length (Full)
    len_med_full = df['mean_response_length'].median()
    # Median Length (Stable)
    len_med_stable = df['mean_response_length'].iloc[-50:].median() if len(df) > 50 else df['mean_response_length'].median()
    
    # Exploration Efficiency (Stable)
    # Image B4: SR 0.0583, LenStable 178.6, Exp 0.3323
    # 0.0583 / 178.6 * 1000 = 0.326.
    # Maybe (Success / Length).mean() * 1000 ?
    df['eta_raw'] = (df['ema_success_rate'] / df['mean_response_length']) * 1000
    exp_stable = df['eta_raw'].iloc[-50:].mean() if len(df) > 50 else df['eta_raw'].mean()
    
    # Hallucination Rate (Full and Stable)
    hall_full = df['hallucination_rate'].mean()
    hall_stable = df['hallucination_rate'].iloc[-50:].mean() if len(df) > 50 else df['hallucination_rate'].mean()
    
    # KL (Full and Stable)
    kl_full = df['kl_div'].mean()
    kl_stable = df['kl_div'].iloc[-50:].mean() if len(df) > 50 else df['kl_div'].mean()
    
    return {
        "StableSuccess": ema_stable,
        "PeakSuccess": ema_peak,
        "LenMedFull": len_med_full,
        "LenMedStable": len_med_stable,
        "ExpStable": exp_stable,
        "HallFull": hall_full,
        "HallStable": hall_stable,
        "KLFull": kl_full,
        "KLStable": kl_stable
    }

# Table 5-3: N=4
print("--- Data for Table 5-3 (N=4) ---")
print(f"{'Config':<10} | {'SR St':<8} | {'SR Pk':<8} | {'Len Fl':<8} | {'Len St':<8} | {'Hall Fl':<8} | {'KL Fl':<8} | {'Exp St':<8}")
for b, f in files_n4.items():
    res = analyze_file(os.path.join(log_dir, f))
    if res:
        print(f"{b:<10} | {res['StableSuccess']:<8.4f} | {res['PeakSuccess']:<8.4f} | {res['LenMedFull']:<8.1f} | {res['LenMedStable']:<8.1f} | {res['HallFull']:<8.3f} | {res['KLFull']:<8.2f} | {res['ExpStable']:<8.3f}")

# Table 5-4: MIX
print("\n--- Data for Table 5-4 (N=3,4,5,6) ---")
print(f"{'Config':<10} | {'SR St':<8} | {'SR Pk':<8} | {'Len Fl':<8} | {'Len St':<8} | {'Hall Fl':<8} | {'KL Fl':<8} | {'Exp St':<8}")
for b, f in files_mix.items():
    res = analyze_file(os.path.join(log_dir, f))
    if res:
        print(f"{b:<10} | {res['StableSuccess']:<8.4f} | {res['PeakSuccess']:<8.4f} | {res['LenMedFull']:<8.1f} | {res['LenMedStable']:<8.1f} | {res['HallFull']:<8.3f} | {res['KLFull']:<8.2f} | {res['ExpStable']:<8.3f}")


