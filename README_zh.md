# LAGRPO：面向小型语言模型的逻辑与长度感知组相对策略优化

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-informational.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![TRL 0.11+](https://img.shields.io/badge/TRL-0.11%2B-purple.svg)](https://github.com/huggingface/trl)

[ [English](README.md) | [中文](README_zh.md) ]

## 摘要

小型语言模型（SLM）在受限算力环境（如单张 NVIDIA RTX 4090 24GB 显卡）下开展复杂逻辑推理任务的强化学习对齐时，面临显著的优化瓶颈。传统的近端策略优化（PPO）因引入价值网络（Critic）而存在极高方差与显存开销；组相对策略优化（GRPO）虽降低了显存需求，但在稀疏奖励下易遭遇**长度偏好导致的词元膨胀**、**连续 Dense 奖励引起的奖励劫持**以及**阶段切换时的梯度冲击**。

本项目提出 **LAGRPO (Logic-Aware & Length-Aware Group Relative Policy Optimization)** 框架，包含以下核心机制：
1. **长度感知优势归一化**：解耦 Token 生成长度对策略梯度的隐式加权，消除词元膨胀偏好。
2. **平滑奖励退火**：采用基于 EMA 成功率的 Sigmoid 函数，实现 Dense 引导奖励向 Binary 真值判定奖励的平滑过渡，避免梯度冲击。
3. **非对称优势截断**：限制组内双峰稀疏奖励产生的离群优势值，稳定梯度更新。

在 Arithmetic-24 数学推理基准测试中，LAGRPO 相比 Vanilla GRPO 实现**稳定段 EMA 成功率提升 55.5%**，**探索效率 ($\eta$) 提升 63.1%**，且保持极低的策略漂移（KL 方差为 $0.21$，远低于 PPO 的 $1139.21$）。

---

## 算法原理与数学建模

```
                          +------------------------------------------+
                          |        策略网络 \pi_\theta                |
                          +------------------------------------------+
                                               |
                                     组采样 (Group Size G)
                                               v
+----------------------------------------------------------------------------------------------+
| 复合多阶段过程监督沙盒 (PRM Sandbox)                                                         |
|                                                                                              |
| [阶段 1: 格式护栏] ----(通过)----> [阶段 2: AST 过程监督] ----(通过)----> [阶段 3: 终态评判]  |
| 校验 <think> 标签                       解析中间逻辑步 A=B                  Sigmoid 退火     |
| 违规直接惩罚 R = -1.0                    步骤奖励 R = +0.1                  Dense / Binary   |
+----------------------------------------------------------------------------------------------+
                                               |
                                        奖励向量 r_i
                                               v
+----------------------------------------------------------------------------------------------+
| 优势归一化与裁剪引擎                                                                         |
|                                                                                              |
| 1. 组内 Z-score 归一化:          A_i = (r_i - \bar{r}) / (\sigma_r + \epsilon)               |
| 2. 长度感知重缩放:               A_i^{len} = A_i \cdot (\bar{L}_{group} / L_i)               |
| 3. 非对称截断:                   A_i^{clip} = \text{clamp}(A_i^{len}, -c, +c)                |
+----------------------------------------------------------------------------------------------+
                                               |
                                        策略梯度 Loss
                                               v
                          +------------------------------------------+
                          |        更新策略参数 \theta                |
                          +------------------------------------------+
```

### 1. 长度感知优势归一化 (Length-Aware Advantage Normalization)

标准 GRPO 目标函数定义如下：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right]$$

将标量优势 $\hat{A}_i$ 均匀广播至响应 $i$ 的全量 $|o_i|$ 个 Token 上，导致总梯度贡献与 $|o_i| \cdot \hat{A}_i$ 成正比。在稀疏奖励场景下，这会引入系统性的**长度偏好**，激励模型生成冗长序列以获取更大的梯度权重。

LAGRPO 引入长度感知重缩放：

$$\hat{A}_i^{\text{len}} = \hat{A}_i \cdot \frac{\bar{L}_{\text{group}}}{L_i}, \quad \text{其中 } \bar{L}_{\text{group}} = \frac{1}{G} \sum_{j=1}^G L_j$$

该项消除了响应长度 $L_i$ 对整体参数更新量的隐式影响。

### 2. 平滑奖励退火 (Smooth Reward Annealing)

为平衡早期探索引导与后期逻辑真值判定，避免硬切换引起的梯度冲击，LAGRPO 对 Dense 距离奖励 $r^{\text{dense}}$ 和 Binary 真值奖励 $r^{\text{binary}}$ 进行连续插值：

$$r = (1 - \alpha) \cdot r^{\text{dense}} + \alpha \cdot r^{\text{binary}}$$

退火系数 $\alpha$ 基于 EMA 成功率 $\bar{s}$ 进行 Sigmoid 参数化：

$$\alpha = \sigma\left(\frac{\bar{s} - \tau}{T}\right) = \frac{1}{1 + e^{-(\bar{s} - \tau) / T}}$$

其中切换阈值设定为 $\tau = 0.10$，温度参数 $T = 0.02$。

### 3. 非对称优势截断 (Asymmetric Advantage Clipping)

在小组规模（$G \in \{4, 8\}$）的稀疏奖励环境中，双峰奖励分布容易引发 Z-score 优势值异常剧变。LAGRPO 在 $c = 3.0$ 倍标准差处进行对称截断：

$$\hat{A}_i^{\text{clip}} = \text{clamp}(\hat{A}_i, -c, +c)$$

### 4. 过程监督与可解性课程 (PRM & Solvable Curriculum)

- **AST 过程沙盒**：通过 Python AST 解析中间推导步骤（如 `A = B`）。若查出算术伪造赋予负惩罚 $R = -1.0$，合法推导赋予步奖励 $R = +0.1$。
- **可解性预过滤**：在训练前剔除无解题目配置，消除组内样本全失败导致的零方差无效更新步。

---

## 实验结果与分析

所有实验均在单张 NVIDIA RTX 4090 GPU（24GB VRAM）上运行，Effective Batch Size 严格对齐为 $16$。

### 1. 基线对比（PPO vs. GRPO）

| 算法 | 平均成功率 (最后 20 步) | KL 散度均值 | KL 散度方差 | Levene 方差齐性检验 ($p$-value) |
| :--- | :---: | :---: | :---: | :---: |
| **PPO** | 0.94% ± 2.29% | 35.53 | 1139.21 | — |
| **GRPO ($G=4$)** | 15.63% ± 11.38% | 3.10 | 1.31 | $p < 0.001$ |
| **GRPO ($G=16$)** | 14.38% ± 16.61% | 3.10 | 0.21 | $p < 0.001$ |

> 双样本 $t$ 检验显示 GRPO 性能显著优于 PPO ($p < 0.01$)。Levene 方差齐性检验证明 GRPO 策略方差显著低于 PPO ($p < 0.001$)。

### 2. 消融实验一（$N=4$ 专项任务）

消融配置划分：
- **B0**：Vanilla GRPO ($G=8$)
- **B1**：B0 + 长度惩罚
- **B2**：B1 + 奖励退火
- **B3**：B2 + 优势截断
- **B4**：Full LAGRPO（完整机制）

| 配置 | EMA 成功率 (稳定段) | EMA 成功率 (峰值) | 长度中位数 (tokens) | 幻觉率 | 探索效率 $\eta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B0** | 0.1483 | 0.1709 | 194.5 | 0.400 | 0.794 |
| **B1** | 0.0668 | 0.0975 | 222.6 | 0.326 | 0.319 |
| **B2** | 0.0798 | 0.1061 | 249.9 | 0.275 | 0.348 |
| **B3** | 0.0754 | 0.1124 | 267.2 | 0.319 | 0.309 |
| **B4** | **0.1943** | **0.2166** | 228.5 | 0.363 | **0.856** |

### 3. 消融实验二（多难度混合 $N \in \{3,4,5,6\}$）

| 配置 | EMA 成功率 (稳定段) | 长度中位数 (tokens) | 幻觉率 | 探索效率 $\eta$ |
| :--- | :---: | :---: | :---: | :---: |
| **B0 (Vanilla GRPO)** | 0.0375 | 190.1 | 0.315 | 0.204 |
| **B4 (Full LAGRPO)** | **0.0583** | **188.3** | 0.457 | **0.332** |

> **探索效率指标 ($\eta$)**：定义为 $\eta = \frac{\text{EMA 成功率}}{\text{生成长度}} \times 10^3$，衡量模型每生成 1000 个 Token 换取的有效成功率提升。

---

## 环境安装与快速开始

### 1. 依赖安装

```bash
git clone https://github.com/KardeniaPoyu/LAGRPO.git
cd LAGRPO
pip install -r requirements.txt
```

### 2. 数据准备与 SFT 预热

```bash
# 生成多难度 Arithmetic-24 数据集
python data_gen_multi.py

# 训练基线 SFT 模型
python train_sft.py --model-name Qwen/Qwen2.5-0.5B-Instruct --output-dir saved_models/sft_final
```

### 3. 训练与自动化流水线

```bash
# 运行 Vanilla GRPO (B0)
python train_grpo.py --ablation B0 --group-size 8 --max-steps 200

# 运行 Full LAGRPO (B4)
python train_grpo.py --ablation B4 --group-size 8 --max-steps 200

# 执行论文全量自动化消融流水线 (B0 至 B4)
python scripts/run_paper_ablations.py --max-steps 200 --group-size 32
```

### 4. 多随机种子复现流水线 (统计显著性验证)

复现 NeurIPS/ICLR 级包含置信区间带的多随机种子实验：

```bash
# 自动化多种子复现流水线 (Seeds: 42, 2026, 999)
bash scripts/run_multi_seed.sh

# 运行完整综合评估套件
bash scripts/run_integrated_suite.sh
```

### 5. 评估与可视化

```bash
# 在测试集上评估已训练检查点
python evaluate.py --models saved_models/ablations/grpo_b4_G32_final

# 生成高分辨率学术统计图表 (在 plots/ 中)
python scripts/archive/generate_lagrpo_plots.py
```

---

## 目录结构

```
.
├── train_grpo.py              # GRPO 与 LAGRPO 核心训练引擎
├── train_ppo.py               # PPO 基线实现
├── train_sft.py               # 监督微调流水线
├── env.py                     # 任务环境与 AST 过程校验沙盒
├── model_utils.py             # 模型加载与 LoRA 配置
├── evaluate.py                # 测试集评估脚本
├── data_gen_multi.py          # 多难度数据集生成器
├── scripts/                   # 自动化流水线与多种子复现脚本
│   ├── run_paper_ablations.py # 消融实验流水线 (B0-B4)
│   ├── run_multi_seed.sh      # 多随机种子自动化复现流水线 (Seeds: 42, 2026, 999)
│   ├── run_integrated_suite.sh# 综合评估套件
│   └── archive/               # 辅助分析与作图脚本
├── docs/                      # 理论推导与 LaTeX 论文源码
├── plots/                     # 高分辨率学术图表 (PNG, SVG, PDF)
├── requirements.txt           # 依赖配置文件
└── LICENSE                    # MIT 许可证
```

---

## 引用格式

```bibtex
@misc{zhou2026lagrpo,
  author = {Yirong Zhou},
  title = {LAGRPO: Logic-Aware & Length-Aware Group Relative Policy Optimization for Small Language Models},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KardeniaPoyu/LAGRPO}}
}
```

## 许可证

本项目遵循 [MIT 许可证](LICENSE)。
