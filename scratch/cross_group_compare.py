import pandas as pd
import numpy as np

def cross_group_comparison():
    # 数据源：LAGRPO (G=8) vs 标准 GRPO (G=16)
    lagrpo = pd.read_csv('logs/grpo_ablation_B4_FINAL_G8_metrics.csv')
    std_g16 = pd.read_csv('logs/grpo_G16_metrics.csv')
    
    def get_summary(df, label):
        # 取最末尾的稳定区间（最后15步）
        last_window = df.iloc[-15:]
        sr = last_window['success_rate'].mean()
        length = last_window['mean_response_length'].mean()
        eta = (last_window['success_rate'] / (last_window['mean_response_length'] + 1e-5) * 1000).mean()
        
        return {
            'Success Rate (%)': sr * 100,
            'Avg Length (Tokens)': length,
            'Efficiency (eta)': eta
        }

    report = pd.DataFrame({
        'LAGRPO (Full, G=8)': get_summary(lagrpo, 'LAGRPO'),
        'Standard GRPO (G=16)': get_summary(std_g16, 'Std G16')
    })
    
    print("=== Cross-Group Selection Comparison (LAGRPO G8 vs Std G16) ===")
    print(report.round(3))
    
    # 效率比分析
    rel_eff = report.loc['Efficiency (eta)', 'LAGRPO (Full, G=8)'] / report.loc['Efficiency (eta)', 'Standard GRPO (G=16)']
    print(f"\n[Conclusion] LAGRPO (G=8) achieves {rel_eff:.2f}x the efficiency of Standard GRPO (G=16).")

if __name__ == "__main__":
    cross_group_comparison()
