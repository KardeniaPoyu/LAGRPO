import pandas as pd
import os

def check_length_discrepancy():
    files = {
        'B0 (Ablation Baseline, G=8)': 'logs/grpo_ablation_B0_G8_metrics.csv',
        'Std GRPO (G=8)': 'logs/grpo_G8_metrics.csv',
        'Std GRPO (G=16)': 'logs/grpo_G16_metrics.csv',
        'Baseline (General)': 'logs/grpo_baseline_metrics.csv'
    }
    
    results = []
    for label, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            results.append({
                'Experiment': label,
                'Start Length (Avg 0-5)': df['mean_response_length'].iloc[:5].mean(),
                'Mid Length (Avg 40-50)': df['mean_response_length'].iloc[40:50].mean() if len(df) > 50 else None,
                'Success Rate (End)': df['success_rate'].iloc[-1],
                'Step Count': len(df)
            })
    
    print(pd.DataFrame(results))

if __name__ == "__main__":
    check_length_discrepancy()
