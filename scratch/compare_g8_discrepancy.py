import pandas as pd
import numpy as np

def compare_g8_discrepancy():
    # 数据加载
    scaling_g8 = pd.read_csv('logs/grpo_G8_metrics.csv')
    ablation_b0 = pd.read_csv('logs/grpo_ablation_B0_G8_metrics.csv')
    
    def get_profile(df, name):
        # 初始状态 (前 3 步平均)
        start = df.iloc[:3]
        # 稳定状态 (后 10 步平均)
        end = df.iloc[-10:]
        
        return {
            'Name': name,
            'Step0_Length': df['mean_response_length'].iloc[0],
            'Step0_SR (%)': df['success_rate'].iloc[0] * 100,
            'Init_Entropy': df['policy_entropy'].iloc[0],
            'Terminal_Length': end['mean_response_length'].mean(),
            'Terminal_SR (%)': end['success_rate'].mean() * 100,
            'Length_Growth': end['mean_response_length'].mean() - start['mean_response_length'].mean(),
            'Max_KL': df['kl_div'].max() if 'kl_div' in df.columns else (df['kl_ref'].max() if 'kl_ref' in df.columns else 0)
        }

    report = pd.DataFrame([
        get_profile(scaling_g8, 'Standard Scaling (G=8)'),
        get_profile(ablation_b0, 'Ablation Baseline (B0)')
    ])
    
    print("=== G=8 Discrepancy Analysis Report ===")
    print(report.round(2).to_string(index=False))
    
    # 探查成功率曲线的相关性
    print("\n--- 动态演进对比 ---")
    print(f"Standard G8 Mean Length (Full Trajectory): {scaling_g8['mean_response_length'].mean():.2f}")
    print(f"Ablation B0 Mean Length (Full Trajectory): {ablation_b0['mean_response_length'].mean():.2f}")

if __name__ == "__main__":
    compare_g8_discrepancy()
