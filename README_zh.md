# LAGRPO：面向小型语言模型的逻辑与长度感知组相对策略优化

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![TRL 0.11+](https://img.shields.io/badge/TRL-0.11%2B-purple.svg)](https://github.com/huggingface/trl)
[![Model: Qwen2.5-7B](https://img.shields.io/badge/Model-Qwen2.5--7B-green.svg)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

[ **[English README](README.md)** | **[中文文档](README_zh.md)** ]

---

## 📌 项目概述

**LAGRPO (Logic-Aware & Length-Aware Group Relative Policy Optimization)** 是一种专为受限单卡硬件环境（如单张 NVIDIA RTX 4090 24GB VRAM）设计的高效强化学习（RL）推理对齐框架。

在复杂、稀疏奖励的逻辑推理任务（如 24 点游戏 *Arithmetic-24*）中，直接应用标准 PPO 或原生 GRPO 容易遭遇以下策略崩溃问题：
1. **无效词元膨胀 (Ineffective Token Bloat)**：模型学会生成冗长的“虚假思维链”，试图通过延长序列来最大化 Token 级别的累积梯度。
2. **奖励劫持 (Reward Hacking)**：连续的 Dense 奖励导致模型收敛至局部最优（如输出接近 24 但逻辑错误的表达式）。
3. **梯度冲击 (Gradient Shock)**：硬切换 Dense 与 Binary 奖励阶段引发优势方差剧烈震荡，导致策略崩溃。

**LAGRPO** 通过三大核心算法机制与多阶段过程监督沙盒（PRM），在消费级显卡算力下实现了相较于 Vanilla GRPO **+55.5% 的稳定段成功率提升** 以及 **+63.1% 的探索效率 ($\eta$) 优化**。

---

## 🚀 核心算法机制

```
                                 +---------------------------------------+
                                 |         SLM 策略网络 (Qwen2.5-7B)     |
                                 +---------------------------------------+
                                                     |
                                            组采样 (Group Size G)
                                                     v
+---------------------------------------------------------------------------------------------------+
| 复合多阶段奖励与 PRM 过滤沙盒                                                                     |
|                                                                                                   |
|  [阶段 1: 格式护栏] ------(通过)-----> [阶段 2: 过程监督 PRM] ------(通过)-----> [阶段 3: 终态校验]
|  严格检查 <think> 标签                 AST 二次验证中间等式 (A=B)               Sigmoid 动态退火
|  违规直接截断 R=-1.0                   幻觉作弊扣分 R=-1.0 / 步奖励 R=+0.1       Dense -> Binary 奖励
+---------------------------------------------------------------------------------------------------+
                                                     |
                                               奖励向量 r_i
                                                     v
+---------------------------------------------------------------------------------------------------+
| LAGRPO 优势估计引擎                                                                               |
|                                                                                                   |
|  1. 组内 Z-score 归一化:         A_i = (r_i - mean(r)) / std(r)                                   |
|  2. 长度感知优势归一化:          A_i^{len} = A_i * (L_group / L_i)                                |
|  3. 非对称优势截断:              A_i^{clip} = clamp(A_i^{len}, -3.0, +3.0)                        |
+---------------------------------------------------------------------------------------------------+
                                                     |
                                            策略梯度更新
                                                     v
                                 +---------------------------------------+
                                 |             更新后的策略参数          |
                                 +---------------------------------------+
```

### 1. ⚡ 长度感知优势归一化 (Length-Aware Advantage Normalization)
标准 GRPO 将标量优势值 $\hat{A}_i$ 均匀广播到响应 $i$ 的每一个 Token 上，导致总梯度贡献与序列长度 $L_i$ 成正比，引入了**长度偏好**。LAGRPO 通过乘以组内平均长度与个体长度之比，使梯度更新量与序列长度解耦：

$$\hat{A}_i^{\text{len}} = \hat{A}_i \cdot \frac{\bar{L}_{\text{group}}}{L_i}$$

其中 $\bar{L}_{\text{group}} = \frac{1}{G} \sum_{j=1}^{G} L_j$。长序列获得更小的单 Token 优势量，彻底根治无效词元膨胀。

### 2. 🔄 Sigmoid 平滑奖励退火 (Smooth Reward Annealing)
为消除由探索阶段（Dense 距离奖励）向利用阶段（Binary 逻辑真值奖励）切换时的**梯度冲击**，LAGRPO 引入基于 EMA 成功率 $\bar{s}$ 的连续可微插值：

$$r = (1 - \alpha) \cdot r^{\text{dense}} + \alpha \cdot r^{\text{binary}}$$

退火系数 $\alpha$ 自动调节：

$$\alpha = \sigma\left(\frac{\bar{s} - \tau}{T}\right) = \frac{1}{1 + e^{-(\bar{s} - \tau) / T}}$$

默认设置切换阈值 $\tau=0.10$，温度参数 $T=0.02$。

### 3. 🛡️ 非对称优势截断 (Asymmetric Advantage Clipping)
在组规模 $G$ 较小且奖励极度稀疏的场景下，组内双峰分布（一正多负）会导致归一化优势值异常偏大。LAGRPO 将优势限定在 $3\sigma$ 范围内：

$$\hat{A}_i^{\text{clip}} = \text{clamp}(\hat{A}_i, -c, +c), \quad c = 3.0$$

### 4. 🔍 复合多阶段过滤与过程监督 (PRM)
- **阶段 1（格式护栏）**：强制校验 `<think>...</think>` 闭合标签及字符白名单，违规直接给予最大负惩罚 ($R = -1.0$)。
- **阶段 2（过程监督 PRM）**：基于 PyTorch/Python AST 运行极速安全沙盒，动态解析中间推导 `A = B`。严惩捏造等式的幻觉行为 ($R = -1.0$)，奖励合规推导 ($R = +0.1$)。
- **阶段 3（终态评判）**：校验数字多重集使用规则与精确计算真值 ($R = +4.0$)。

### 5. 🎯 可解性预过滤 (Solvable-Only Curriculum)
离线递归穷举求解器预先剔除数据集中的无解题目，消除组内零方差无效更新步，使**有效学习率提升 1.33 倍**。

---

## 📊 实验验证与结果分析

所有实验均在单张 **NVIDIA RTX 4090 (24GB VRAM)** 显卡上完成，严格控制 Effective Batch Size = 16。

### 1. PPO 与 GRPO 策略漂移对比

| 算法 | 收敛段成功率 | KL 均值 | KL 方差 | Levene 方差齐性检验 | 显存开销 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PPO** | 0.94% ± 2.29% | 35.53 | **1139.21** | — | 高 (需维护 Critic) |
| **GRPO ($G=4$)** | 15.63% ± 11.38% | 3.10 | **1.31** | $p < 0.001$ | 低 (Zero-Critic) |
| **GRPO ($G=16$)** | 14.38% ± 16.61% | 3.10 | **0.21** | $p < 0.001$ | 低 (Zero-Critic) |

> **统计学检验**：双样本 $t$ 检验结果显示 GRPO 收敛速度显著优于 PPO ($p < 0.01$)；Levene 方差齐性检验证明 GRPO 策略更新具备极强确定性 ($p < 0.001$)。

### 2. LAGRPO 消融实验一（$N=4$ 专项任务）

| 机制配置 | EMA 成功率 (稳定段) | EMA 成功率 (峰值) | 长度中位数 (Tokens) | 幻觉率 | 探索效率 $\eta$ (稳定段) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B0 (Vanilla GRPO)** | 0.1483 | 0.1709 | 194.5 | 0.400 | 0.794 |
| **B1 (+ Length Penalty)** | 0.0668 | 0.0975 | 222.6 | 0.326 | 0.319 |
| **B2 (+ Reward Annealing)**| 0.0798 | 0.1061 | 249.9 | 0.275 | 0.348 |
| **B3 (+ Adv Clipping)** | 0.0754 | 0.1124 | 267.2 | 0.319 | 0.309 |
| **B4 (Full LAGRPO)** | **0.1943 (+31.0%)** | **0.2166** | 228.5 | 0.363 | **0.856 (+7.8%)** |

### 3. 消融实验二（多难度混合 $N \in \{3,4,5,6\}$）

| 机制配置 | EMA 成功率 (稳定段) | 长度中位数 (Tokens) | 幻觉率 | 探索效率 $\eta$ (稳定段) |
| :--- | :---: | :---: | :---: | :---: |
| **B0 (Vanilla GRPO)** | 0.0375 | 190.1 | 0.315 | 0.204 |
| **B4 (Full LAGRPO)** | **0.0583 (+55.5%)** | **188.3** | 0.457 | **0.332 (+63.1%)** |

> **探索效率指标 ($\eta$)**：定义为 $\eta = \frac{\text{EMA 成功率}}{\text{生成长度}} \times 10^3$，用以量化模型每生成 1000 个 Token 换取的有效成功率提升。

---

## 🛠️ 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/KardeniaPoyu/SLM-RL-Comparation.git
cd SLM-RL-Comparation

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据生成与 SFT 预热

```bash
# 生成 24 点多难度数据集
python data_gen_multi.py

# 预训练 SFT 模型
python train_sft.py --model-name Qwen/Qwen2.5-0.5B-Instruct --output-dir saved_models/sft_final
```

### 3. 运行训练

```bash
# 运行 Vanilla GRPO (B0)
python train_grpo.py --ablation B0 --group-size 8 --max-steps 200

# 运行 Full LAGRPO (B4)
python train_grpo.py --ablation B4 --group-size 8 --max-steps 200

# 运行 PPO 基线
python train_ppo.py --max-steps 200
```

### 4. 自动化消融与评测流水线

```bash
# 试跑消融流水线
python run_paper_ablations.py --dry-run

# 执行完整 B0 -> B4 消融实验与测试集评估
python run_paper_ablations.py --max-steps 200 --group-size 32
```

### 5. 可视化与统计分析

```bash
# 生成高分辨率学术统计图表
python generate_lagrpo_plots.py
python analyze_ablation_v2.py
```

---

## 📄 许可证与引用

本项目采用 **MIT 许可证**。

如果您在研究或工作中参考了本项目，请按如下格式引用：

```bibtex
@misc{zhou2026lagrpo,
  author = {Yirong Zhou},
  title = {LAGRPO: Logic-Aware & Length-Aware Group Relative Policy Optimization for Small Language Models},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KardeniaPoyu/SLM-RL-Comparation}}
}
```
