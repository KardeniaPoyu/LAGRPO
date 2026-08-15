import pandas as pd
import numpy as np

def compare_methods():
    # 加载两个核心对比组的数据 (G=8 受限资源场景)
    b0 = pd.read_csv('logs/grpo_ablation_B0_G8_metrics.csv')
    b4 = pd.read_csv('logs/grpo_ablation_B4_FINAL_G8_metrics.csv')
    
    def get_summary(df):
        # 取最后 20 步的平均值作为收敛状态
        last_window = df.iloc[-20:]
        sr = last_window['success_rate'].mean()
        length = last_window['mean_response_length'].mean()
        # 计算效率指标 eta = (SR / Length) * 1000
        eta = (last_window['success_rate'] / (last_window['mean_response_length'] + 1e-5) * 1000).mean()
        kl = last_window.get('kl_div', last_window.get('kl_ref', [0])).iloc[-1]
        grad_var = last_window['grad_second_moment'].mean()
        
        return {
            'Success Rate (%)': sr * 100,
            'Avg Length (Tokens)': length,
            'Efficiency (eta)': eta,
            'KL Deviation': kl,
            'Grad Variance (log10)': np.log10(grad_var + 1e-12)
        }

    report = pd.DataFrame({
        'Standard GRPO (B0)': get_summary(b0),
        'LAGRPO Full (B4)': get_summary(b4)
    })
    
    # 计算增益百分比
    report['Gain (%)'] = (report['LAGRPO Full (B4)'] - report['Standard GRPO (B0)']) / (report['Standard GRPO (B0)'] + 1e-9) * 100
    
    print("=== LAGRPO vs. Standard GRPO Comparison Report ===")
    print(report.round(3))
    
    print("\n--- 关键结论 ---")
    if report.loc['Efficiency (eta)', 'Gain (%)'] > 0:
        print(f"[Efficiency] LAGRPO is {report.loc['Efficiency (eta)', 'Gain (%)']:.1f}% MORE EFFICIENT than Standard GRPO.")
    if report.loc['Avg Length (Tokens)', 'Gain (%)'] < 0:
        print(f"[Compactness] LAGRPO reduced response length by {abs(report.loc['Avg Length (Tokens)', 'Gain (%)']):.1f}%.")

if __name__ == "__main__":
    compare_methods()
