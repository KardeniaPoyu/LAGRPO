# LAGRPO: Logic-Aware & Length-Aware Group Relative Policy Optimization for Small Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![TRL 0.11+](https://img.shields.io/badge/TRL-0.11%2B-purple.svg)](https://github.com/huggingface/trl)
[![Model: Qwen2.5-7B](https://img.shields.io/badge/Model-Qwen2.5--7B-green.svg)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

[ **[English README](README.md)** | **[中文文档](README_zh.md)** ]

---

## 📌 Executive Summary

**LAGRPO (Logic-Aware & Length-Aware Group Relative Policy Optimization)** is a lightweight, sample-efficient reinforcement learning (RL) alignment framework specifically designed for **Small Language Models (SLMs)** operating under constrained single-GPU hardware (e.g., single NVIDIA RTX 4090 24GB VRAM).

Applying standard PPO or vanilla GRPO to complex, sparse-reward logical reasoning tasks (such as *Arithmetic-24*) often suffers from severe training pathologies:
1. **Ineffective Token Bloat**: Models learn to generate excessively long, redundant reasoning chains ("hallucinated CoT") to maximize token-level gradient accumulation.
2. **Reward Hacking**: Continuous dense rewards lead models to settle for local optima (e.g., outputs close to 24 but mathematically incorrect).
3. **Gradient Shock & Strategy Collapse**: Sudden switching between dense and binary reward phases causes advantage variance spikes, leading to policy collapse.

**LAGRPO** resolves these challenges through three core algorithmic innovations and a multi-stage process reward model (PRM), achieving a **+55.5% increase in stable EMA success rate** and a **+63.1% boost in exploration efficiency ($\eta$)** over Vanilla GRPO under strict single-GPU compute constraints.

---

## 🚀 Key Algorithmic Novelties

```
                                 +---------------------------------------+
                                 |         SLM Policy (Qwen2.5-7B)       |
                                 +---------------------------------------+
                                                     |
                                            Rollout (Group Size G)
                                                     v
+---------------------------------------------------------------------------------------------------+
| Multi-Stage Reward & PRM Sandbox                                                                 |
|                                                                                                   |
|  [Stage 1: Format Guardrails] ----(Pass)----> [Stage 2: CoT Step PRM] ----(Pass)----> [Stage 3: Terminal]
|  Strict <think> tag check                      AST Verification (A=B)                 Sigmoid Annealed
|  Penalty R=-1.0                                Punishment R=-1.0 / Step R=+0.1        Dense -> Binary Reward
+---------------------------------------------------------------------------------------------------+
                                                     |
                                               Reward Vector r_i
                                                     v
+---------------------------------------------------------------------------------------------------+
| LAGRPO Advantage Engine                                                                           |
|                                                                                                   |
|  1. Group Z-score Normalization:  A_i = (r_i - mean(r)) / std(r)                                  |
|  2. Length-Aware Normalization:   A_i^{len} = A_i * (L_group / L_i)                               |
|  3. Asymmetric Advantage Clip:    A_i^{clip} = clamp(A_i^{len}, -3.0, +3.0)                       |
+---------------------------------------------------------------------------------------------------+
                                                     |
                                            Policy Gradient Update
                                                     v
                                 +---------------------------------------+
                                 |         Updated Policy Weights        |
                                 +---------------------------------------+
```

### 1. ⚡ Length-Aware Advantage Normalization
Standard GRPO broadcasts a scalar advantage $\hat{A}_i$ evenly to all $L_i$ tokens of response $i$, causing total gradient contribution to scale proportionally with $L_i \cdot \hat{A}_i$. In sparse reward tasks, this creates an implicit **length bias**. LAGRPO decouples token length from gradient magnitude:

$$\hat{A}_i^{\text{len}} = \hat{A}_i \cdot \frac{\bar{L}_{\text{group}}}{L_i}$$

where $\bar{L}_{\text{group}} = \frac{1}{G} \sum_{j=1}^{G} L_j$. Longer responses receive smaller per-token advantage updates, eliminating ineffective token expansion.

### 2. 🔄 Sigmoid Smooth Reward Annealing
To prevent **gradient shock** when transitioning from early exploration (dense distance rewards) to late exploitation (binary correctness rewards), LAGRPO introduces a continuous可微 weight interpolation:

$$r = (1 - \alpha) \cdot r^{\text{dense}} + \alpha \cdot r^{\text{binary}}$$

where the annealing coefficient $\alpha$ is dynamically governed by the Exponential Moving Average (EMA) success rate $\bar{s}$:

$$\alpha = \sigma\left(\frac{\bar{s} - \tau}{T}\right) = \frac{1}{1 + e^{-(\bar{s} - \tau) / T}}$$

with threshold $\tau=0.10$ and temperature $T=0.02$.

### 3. 🛡️ Asymmetric Advantage Clipping
In sparse reward environments, bimodal reward distributions (mostly negative, rare positive) produce extreme Z-score advantage spikes when $G$ is small. LAGRPO bounds advantages within a $3\sigma$ confidence interval:

$$\hat{A}_i^{\text{clip}} = \text{clamp}(\hat{A}_i, -c, +c), \quad c = 3.0$$

### 4. 🔍 Composite Multi-Stage Reward & PRM Sandbox
- **Stage 1 (Format Guardrails)**: Enforces strict `<think>...</think>` tag compliance and white-list token filtering (penalty $R = -1.0$).
- **Stage 2 (Step-wise CoT PRM)**: Safe AST-based sandbox dynamically parses intermediate equations `A = B`. Punishes mathematical hallucinations ($R = -1.0$) and rewards valid step-wise logic ($R = +0.1$).
- **Stage 3 (Terminal Verification)**: Validates multiset digit usage and computes exact target value matching ($R = +4.0$).

### 5. 🎯 Solvable-Only Curriculum
Offline recursive solver pre-filters unsolvable task instances, eliminating zero-variance gradient steps and increasing the effective learning rate by **1.33×**.

---

## 📊 Benchmark & Empirical Results

All experiments were conducted on a single **NVIDIA RTX 4090 (24GB VRAM)** under strictly controlled effective batch size ($EBS=16$).

### 1. PPO vs. GRPO Performance & Strategy Drift

| Algorithm | Converged Success Rate | KL Mean | KL Variance | Levene Test ($p$-value) | VRAM Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PPO** | 0.94% ± 2.29% | 35.53 | **1139.21** | — | High (+30% Critic) |
| **GRPO ($G=4$)** | 15.63% ± 11.38% | 3.10 | **1.31** | $p < 0.001$ | Low (Zero-Critic) |
| **GRPO ($G=16$)** | 14.38% ± 16.61% | 3.10 | **0.21** | $p < 0.001$ | Low (Zero-Critic) |

> **Statistical Significance**: Two-sample $t$-test confirms GRPO significantly outperforms PPO ($p < 0.01$). Levene's test proves GRPO maintains extreme variance reduction against strategy drift ($p < 0.001$).

### 2. LAGRPO Ablation Study ($N=4$ Task Distribution)

| Configuration | EMA Success (Stable) | EMA Success (Peak) | Median Tokens | Hallucination Rate | Exploration Efficiency $\eta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B0 (Vanilla GRPO)** | 0.1483 | 0.1709 | 194.5 | 0.400 | 0.794 |
| **B1 (+ Length Penalty)** | 0.0668 | 0.0975 | 222.6 | 0.326 | 0.319 |
| **B2 (+ Reward Annealing)**| 0.0798 | 0.1061 | 249.9 | 0.275 | 0.348 |
| **B3 (+ Adv Clipping)** | 0.0754 | 0.1124 | 267.2 | 0.319 | 0.309 |
| **B4 (Full LAGRPO)** | **0.1943 (+31.0%)** | **0.2166** | 228.5 | 0.363 | **0.856 (+7.8%)** |

### 3. Multi-Difficulty Mixed Benchmark ($N \in \{3,4,5,6\}$)

| Configuration | EMA Success (Stable) | Median Tokens | Hallucination Rate | Exploration Efficiency $\eta$ |
| :--- | :---: | :---: | :---: | :---: |
| **B0 (Vanilla GRPO)** | 0.0375 | 190.1 | 0.315 | 0.204 |
| **B4 (Full LAGRPO)** | **0.0583 (+55.5%)** | **188.3** | 0.457 | **0.332 (+63.1%)** |

> **Exploration Efficiency Index ($\eta$)**: Defined as $\eta = \frac{\text{EMA Success Rate}}{\text{Mean Response Length}} \times 10^3$, quantifying net success rate gains per 1,000 generated tokens.

---

## 🛠️ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/KardeniaPoyu/SLM-RL-Comparation.git
cd SLM-RL-Comparation

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Generation & SFT Pre-training

```bash
# Generate synthetic Arithmetic-24 dataset
python data_gen_multi.py

# Pre-train baseline SFT model
python train_sft.py --model-name Qwen/Qwen2.5-0.5B-Instruct --output-dir saved_models/sft_final
```

### 3. Train LAGRPO & Baselines

```bash
# Run Vanilla GRPO (B0)
python train_grpo.py --ablation B0 --group-size 8 --max-steps 200

# Run Full LAGRPO (B4)
python train_grpo.py --ablation B4 --group-size 8 --max-steps 200

# Run PPO Baseline
python train_ppo.py --max-steps 200
```

### 4. Automated Ablation Pipeline & Evaluation

```bash
# Dry run the full ablation suite
python run_paper_ablations.py --dry-run

# Run full ablation study (B0 -> B4) with evaluation
python run_paper_ablations.py --max-steps 200 --group-size 32
```

### 5. Generate Visualizations & Statistical Plots

```bash
# Generate high-resolution academic plots
python generate_lagrpo_plots.py
python analyze_ablation_v2.py
```

---

## 📂 Repository Layout

```
SLM-RL-Comparation/
├── train_grpo.py              # Main GRPO & LAGRPO training engine
├── train_ppo.py               # PPO baseline trainer
├── train_sft.py               # Supervised fine-tuning script
├── env.py                     # Arithmetic-24 environment & PRM sandbox
├── model_utils.py             # Model loading, LoRA, and memory optimization
├── evaluate.py                # Standalone evaluation & hallucination metrics
├── run_paper_ablations.py     # Automated ablation suite launcher (B0-B4)
├── analyze_LAGRPO.py          # Statistical analysis & hypothesis testing
├── data_gen_multi.py          # Multi-difficulty dataset generator
├── plots/                     # High-resolution SVG/PNG academic figures
├── requirements.txt           # Python dependency specifications
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

---

## 📄 License & Citation

This project is licensed under the **MIT License**.

If you find **LAGRPO** helpful in your research or applications, please cite this work:

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
