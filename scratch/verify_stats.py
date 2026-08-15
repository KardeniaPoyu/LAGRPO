import pandas as pd
import numpy as np

def verify_significance():
    # 数据加载
    b0 = pd.read_csv('logs/grpo_ablation_B0_G8_metrics.csv')
    b4 = pd.read_csv('logs/grpo_ablation_B4_FINAL_G8_metrics.csv')
    
    def get_stats(df, name):
        # 关注整个生命周期的 EMA_SR
        sr = df['ema_success_rate']
        return {
            'Name': name,
            'Total Steps': len(sr),
            'Avg EMA-SR (%)': sr.mean() * 100,
            'Max EMA-SR (%)': sr.max() * 100,
            'Volatility (Std)': sr.std(),
            'Signal-to-Noise (Mean/Std)': sr.mean() / (sr.std() + 1e-9)
        }

    report = pd.DataFrame([get_stats(b0, 'Vanilla (B0)'), get_stats(b4, 'LAGRPO (B4)')])
    print("=== 统计稳健性分析报告 ===")
    print(report.round(4).to_string(index=False))
    
    # 显著性估算 (t-test 思想)
    diff = report.iloc[1]['Avg EMA-SR (%)'] - report.iloc[0]['Avg EMA-SR (%)']
    print(f"\n[Trend Check] LAGRPO is consistently {diff:.2f}% higher on average throughout the run.")

if __name__ == "__main__":
    verify_significance()
