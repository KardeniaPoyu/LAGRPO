import pandas as pd

def diagnose_same_metric():
    # 数据加载
    std_g8 = pd.read_csv('logs/grpo_G8_metrics.csv')
    abl_b0 = pd.read_csv('logs/grpo_ablation_B0_G8_metrics.csv')
    
    def get_raw_summary(df, name):
        # 统一使用 'success_rate' 而不是 EMA
        sr = df['success_rate']
        return {
            'Name': name,
            'Step0_SR (%)': sr.iloc[0] * 100,
            'Avg_Raw_SR (%)': sr.mean() * 100,
            'Peak_Raw_SR (%)': sr.max() * 100,
            'Init_Length': df['mean_response_length'].iloc[0],
            'Avg_Length': df['mean_response_length'].mean(),
            'Max_KL': df['kl_div'].max()
        }

    comparison = pd.DataFrame([
        get_raw_summary(std_g8, 'Standard Group (G=8)'),
        get_raw_summary(abl_b0, 'Ablation Baseline (B0)')
    ])
    
    print("=== 同口径对比报告 (基于 Raw Success Rate) ===")
    print(comparison.round(3).to_string(index=False))

if __name__ == "__main__":
    diagnose_same_metric()
