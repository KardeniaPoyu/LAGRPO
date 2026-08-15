import pandas as pd

def diagnose_g8_gap():
    # 数据加载
    std_g8 = pd.read_csv('logs/grpo_G8_metrics.csv')
    abl_b0 = pd.read_csv('logs/grpo_ablation_B0_G8_metrics.csv')
    
    def get_summary(df, name):
        return {
            'Name': name,
            'Step0_SR (%)': df['success_rate'].iloc[0] * 100,
            'Step0_Len': df['mean_response_length'].iloc[0],
            'Avg_SR (%)': df['ema_success_rate'].mean() * 100,
            'Peak_SR (%)': df['success_rate'].max() * 100,
            'Final_KL': df['kl_div'].iloc[-1],
            'Terminal_Len': df['mean_response_length'].iloc[-1]
        }

    comparison = pd.DataFrame([
        get_summary(std_g8, 'Standard G=8'),
        get_summary(abl_b0, 'Ablation B0 (G=8)')
    ])
    
    print("=== Diagnostic: Standard G=8 vs Ablation B0 ===")
    print(comparison.to_string(index=False))
    
    # 检查梯度或学习率相关指标 (如果存在)
    for col in ['grad_second_moment', 'policy_entropy']:
        if col in std_g8.columns and col in abl_b0.columns:
            print(f"\n--- {col} Mean Comparison ---")
            print(f"Standard G8: {std_g8[col].mean():.6f}")
            print(f"Ablation B0: {abl_b0[col].mean():.6f}")

if __name__ == "__main__":
    diagnose_g8_gap()
