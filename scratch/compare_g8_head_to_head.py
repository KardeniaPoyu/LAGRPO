import pandas as pd
import numpy as np

def compare_g8_direct():
    # 路径：B0 (Vanilla G8) vs B4 (Full LAGRPO G8)
    b0 = pd.read_csv('logs/grpo_ablation_B0_G8_metrics.csv')
    b4 = pd.read_csv('logs/grpo_ablation_B4_FINAL_G8_metrics.csv')
    
    def get_terminal_performance(df):
        # 取最后 20 步的平均表现
        last_chunk = df.iloc[-20:]
        sr_mean = last_chunk['success_rate'].mean()
        len_mean = last_chunk['mean_response_length'].mean()
        # 探索效率 η
        eta_mean = (last_chunk['success_rate'] / (last_chunk['mean_response_length'] + 1e-5) * 1000).mean()
        # 训练稳定性 (KL 散度)
        kl = last_chunk.get('kl_div', last_chunk.get('kl_ref', [0])).mean()
        
        return {
            'Success Rate (%)': sr_mean * 100,
            'Avg Response Length': len_mean,
            'Efficiency (eta)': eta_mean,
            'Avg KL Drift': kl
        }

    comparison = pd.DataFrame({
        'Vanilla GRPO (G8)': get_terminal_performance(b0),
        'LAGRPO (G8)': get_terminal_performance(b4)
    })
    
    comparison['Improvement (%)'] = (comparison['LAGRPO (G8)'] - comparison['Vanilla GRPO (G8)']) / (comparison['Vanilla GRPO (G8)'] + 1e-9) * 100
    
    print("=== G=8 Head-to-Head: Vanilla vs. LAGRPO ===")
    print(comparison.round(3))
    
    # 轨迹平滑分析（判断稳定性）
    print("\n--- 稳定性分析 (Standard Deviation) ---")
    print(f"Vanilla SR Std: {b0['success_rate'].std():.4f}")
    print(f"LAGRPO SR Std: {b4['success_rate'].std():.4f}")

if __name__ == "__main__":
    compare_g8_direct()
