#!/usr/bin/env python3
"""
run_beta_sweep.py — LAGRPO β (Length Penalty) 敏感度扫参脚本
执行路径: 循环测试不同的 len-adv-beta 值，验证三阶段效应。
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# 获取仓库根目录
REPO_ROOT = Path(__file__).resolve().parent

def default_python_exe() -> str:
    """获取虚拟环境中的 python 解释器"""
    env = os.environ.get("PYTHON_EXE", "").strip()
    if env and Path(env).exists():
        return env
    for candidate in (
        Path(r"D:\venvs\SLM-RL-Comparation\Scripts\python.exe"),
        REPO_ROOT / ".venv_reasoning" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable

def run_cmd(py: str, args: list[str], cwd: Path) -> bool:
    cmd = [py, "-u", *args]
    print("\n" + "=" * 60)
    print(f"RUNNING: {' '.join(cmd)}")
    print("=" * 60)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(cwd), check=False)
        print(f"FINISHED: code={r.returncode} | time={time.time() - t0:.1f}s")
        return r.returncode == 0
    except KeyboardInterrupt:
        print("\n[!] 用户中断。")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="LAGRPO Beta Sensitivity Sweep")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-path", type=str, default="saved_models/sft_final")
    parser.add_argument("--group-size", "-G", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100, help="每组实验的步数")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="saved_models/beta_sweep")
    parser.add_argument("--log-dir", type=str, default="logs/beta_sweep")
    parser.add_argument("--dry-run", action="store_true", help="仅打印不执行")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有结果")
    
    args = parser.parse_args()
    
    # 扫参矩阵: β 值
    # beta_values = [0.0, 0.01, 0.05, 0.15, 0.30, 0.60, 1.20]
    beta_values = [0.0, 0.05, 0.15, 0.30, 0.60] # 精简版，覆盖关键点
    
    # 检查解释器
    py = default_python_exe()
    print(f"Interpreter: {py}")
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for beta in beta_values:
        exp_id = f"beta_{beta:.2f}"
        log_tag = f"grpo_{exp_id}_G{args.group_size}"
        metrics_file = Path(args.log_dir) / f"{log_tag}_metrics.csv"
        
        # 检查是否已跑过
        if metrics_file.exists() and not args.force:
            print(f"[SKIP] {exp_id} 已经存在: {metrics_file}")
            continue

        # 构造训练命令
        # 注意: 我们使用 --ablation B4 作为基础，然后手动覆盖 --len-adv-beta
        train_args = [
            "train_grpo.py",
            "--ablation", "B4",
            "--model-name", args.model_name,
            "--sft-path", args.sft_path,
            "--exp-id", exp_id,
            "--len-adv-beta", str(beta),
            "--group-size", str(args.group_size),
            "--batch-size", str(args.batch_size),
            "--max-steps", str(args.max_steps),
            "--output-dir", args.output_dir,
            "--log-dir", args.log_dir,
        ]

        if args.dry_run:
            print(f"[DRY-RUN] Would run: {' '.join(train_args)}")
        else:
            success = run_cmd(py, train_args, REPO_ROOT)
            if not success:
                print(f"[ERR] Experiment failed for beta={beta}. Stopping sweep.")
                break

    print("\n" + "=" * 60)
    print("  Sweep Complete!")
    print(f"  Logs are in: {args.log_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
