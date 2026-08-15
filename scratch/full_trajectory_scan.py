import pandas as pd
import numpy as np

def deep_ablation_scan():
    # 数据对齐：Vanilla (B0) vs Full LAGRPO (B4)
    files = {
        'Vanilla GRPO': 'logs/grpo_ablation_B0_G8_metrics.csv',
        'LAGRPO (Ours)': 'logs/grpo_ablation_B4_FINAL_G8_metrics.csv'
    }
    
    results = {}
    for name, path in files.items():
        df = pd.read_csv(path)
        # 计算每一步的效率 eta
        df['eta'] = (df['ema_success_rate'] / (df['mean_response_length'] + 1e-5)) * 1000
        
        results[name] = {
            'Max EMA Success (%)': df['ema_success_rate'].max() * 100,
            'Step of Max SR': df['ema_success_rate'].idxmax(),
            'Mean Response Length (overall)': df['mean_response_length'].mean(),
            'Integrated Efficiency (AUC-eta)': df['eta'].sum(), # 整体探索效能
            'Stability (EMA-SR Variance)': df['ema_success_rate'].iloc[-50:].std() # 后期稳定性
        }
    
    report = pd.DataFrame(results).T
    print("=== 全生命周期效能对比报告 ===")
    print(report.round(4))
    
    print("\n--- 关键动态解读 ---")
    if results['LAGRPO (Ours)']['Integrated Efficiency (AUC-eta)'] > results['Vanilla GRPO']['Integrated Efficiency (AUC-eta)']:
        gain = (results['LAGRPO (Ours)']['Integrated Efficiency (AUC-eta)'] / results['Vanilla GRPO']['Integrated Efficiency (AUC-eta)'] - 1) * 100
        print(f"[Efficiency] LAGRPO is {gain:.1f}% more efficient in TOTAL exploration power.")
    
    if results['LAGRPO (Ours)']['Mean Response Length (overall)'] < results['Vanilla GRPO']['Mean Response Length (overall)']:
        save = (1 - results['LAGRPO (Ours)']['Mean Response Length (overall)'] / results['Vanilla GRPO']['Mean Response Length (overall)']) * 100
        print(f"[Compactness] LAGRPO saves {save:.1f}% tokens on average throughout training.")

if __name__ == "__main__":
    deep_ablation_scan()
