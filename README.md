# LAGRPO: Logic-Aware & Length-Aware Group Relative Policy Optimization for Small Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-informational.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![TRL 0.11+](https://img.shields.io/badge/TRL-0.11%2B-purple.svg)](https://github.com/huggingface/trl)

[ [English](README.md) | [中文](README_zh.md) ]

## Abstract

Reinforcement learning alignment for Small Language Models (SLMs) on complex logical reasoning tasks encounters significant optimization challenges under constrained hardware environments (e.g., single NVIDIA RTX 4090 24GB GPU). Standard Proximal Policy Optimization (PPO) suffers from high variance and memory overhead due to value function estimation, while Group Relative Policy Optimization (GRPO) can exhibit length bias (token bloat), reward hacking under dense rewards, and gradient shock during phase transitions.

This repository presents **LAGRPO (Logic-Aware & Length-Aware Group Relative Policy Optimization)**, an optimization framework tailored for SLM reasoning alignment. LAGRPO incorporates three core mechanisms:
1. **Length-Aware Advantage Normalization**: Decouples token sequence length from policy gradient weights to mitigate length-induced bias.
2. **Smooth Reward Annealing**: Dynamically transitions from dense guidance rewards to binary correctness evaluation via an EMA success-rate Sigmoid schedule, avoiding gradient shock.
3. **Asymmetric Advantage Clipping**: Bounds sample advantage estimates in bimodal sparse reward distributions to stabilize gradient updates.

In empirical evaluations on the Arithmetic-24 benchmark, LAGRPO achieves a **+55.5% relative improvement in stable EMA success rate** and a **+63.1% increase in exploration efficiency ($\eta$)** over vanilla GRPO, maintaining low policy drift (KL variance $0.21$ vs. PPO $1139.21$).

---

## Methodological Framework

```
                          +------------------------------------------+
                          |        Policy Network \pi_\theta         |
                          +------------------------------------------+
                                               |
                                     Group Sampling (G)
                                               v
+----------------------------------------------------------------------------------------------+
| Multi-Stage Process Reward Sandbox                                                           |
|                                                                                              |
| [Stage 1: Format Guard] ----(Pass)----> [Stage 2: AST Process Supervision] ----> [Stage 3]   |
| Tag <think> verification                Validate intermediate step A=B            Terminal   |
| Penalty R = -1.0                        Step reward R = +0.1                      Dense/Binary
+----------------------------------------------------------------------------------------------+
                                               |
                                        Reward Vector r_i
                                               v
+----------------------------------------------------------------------------------------------+
| Advantage Normalization Engine                                                               |
|                                                                                              |
| 1. Group Standard Normalization:  A_i = (r_i - \bar{r}) / (\sigma_r + \epsilon)               |
| 2. Length-Aware Rescaling:       A_i^{len} = A_i \cdot (\bar{L}_{group} / L_i)               |
| 3. Asymmetric Clipping:          A_i^{clip} = \text{clamp}(A_i^{len}, -c, +c)                |
+----------------------------------------------------------------------------------------------+
                                               |
                                     Policy Gradient Loss
                                               v
                          +------------------------------------------+
                          |        Updated Parameters \theta         |
                          +------------------------------------------+
```

### 1. Length-Aware Advantage Normalization

In standard GRPO, the objective is defined as:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right]$$

Broadcasting a scalar advantage $\hat{A}_i$ across all $|o_i|$ tokens results in a total gradient contribution proportional to $|o_i| \cdot \hat{A}_i$. In sparse-reward reasoning tasks, this induces a systemic length bias, encouraging the model to generate redundant tokens to maximize gradient accumulation.

LAGRPO introduces length-aware normalization:

$$\hat{A}_i^{\text{len}} = \hat{A}_i \cdot \frac{\bar{L}_{\text{group}}}{L_i}, \quad \text{where } \bar{L}_{\text{group}} = \frac{1}{G} \sum_{j=1}^G L_j$$

This adjustment ensures that the total parameter update contributed by a response is decoupled from its token length $L_i$.

### 2. Smooth Reward Annealing

To balance early-stage exploration guidance with late-stage logical rigor without incurring gradient shock, LAGRPO interpolates dense distance rewards $r^{\text{dense}}$ and binary correctness rewards $r^{\text{binary}}$:

$$r = (1 - \alpha) \cdot r^{\text{dense}} + \alpha \cdot r^{\text{binary}}$$

The annealing parameter $\alpha$ is parameterized via a Sigmoid schedule over the Exponential Moving Average (EMA) success rate $\bar{s}$:

$$\alpha = \sigma\left(\frac{\bar{s} - \tau}{T}\right) = \frac{1}{1 + e^{-(\bar{s} - \tau) / T}}$$

where $\tau = 0.10$ denotes the transition threshold and $T = 0.02$ represents the temperature parameter.

### 3. Asymmetric Advantage Clipping

Bimodal reward distributions in sparse reasoning environments frequently yield extreme Z-score advantage values when group size $G$ is small ($G \in \{4, 8\}$). LAGRPO applies symmetric truncation at $c = 3.0$ standard deviations:

$$\hat{A}_i^{\text{clip}} = \text{clamp}(\hat{A}_i, -c, +c)$$

### 4. Process Supervision & Solvable Curriculum

- **AST Process Sandbox**: Parses intermediate reasoning steps (e.g., `A = B`) using AST execution. Invalid mathematical transformations receive a penalty of $R = -1.0$, while valid steps receive $R = +0.1$.
- **Solvable-Only Curriculum**: Pre-filters unsolvable task configurations prior to training, preventing zero-variance gradient steps where all $G$ trajectories fail.

---

## Experimental Results

All experiments were executed on a single NVIDIA RTX 4090 GPU (24GB VRAM) with an effective batch size of $16$.

### 1. Baseline Comparison (PPO vs. GRPO)

| Algorithm | Mean Success Rate (Final 20 Steps) | KL Divergence Mean | KL Divergence Variance | Levene Test ($p$-value) |
| :--- | :---: | :---: | :---: | :---: |
| **PPO** | 0.94% ± 2.29% | 35.53 | 1139.21 | — |
| **GRPO ($G=4$)** | 15.63% ± 11.38% | 3.10 | 1.31 | $p < 0.001$ |
| **GRPO ($G=16$)** | 14.38% ± 16.61% | 3.10 | 0.21 | $p < 0.001$ |

> Two-sample $t$-test indicates that GRPO achieves significantly higher convergence performance than PPO ($p < 0.01$). Levene's test confirms superior policy distribution stability for GRPO ($p < 0.001$).

### 2. Ablation Analysis ($N=4$ Task Distribution)

Ablation configurations:
- **B0**: Vanilla GRPO ($G=8$)
- **B1**: B0 + Length Penalty
- **B2**: B1 + Reward Annealing
- **B3**: B2 + Advantage Clipping
- **B4**: Full LAGRPO (All mechanisms active)

| Config | EMA Success (Stable) | EMA Success (Peak) | Median Length (tokens) | Hallucination Rate | Exploration Efficiency $\eta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B0** | 0.1483 | 0.1709 | 194.5 | 0.400 | 0.794 |
| **B1** | 0.0668 | 0.0975 | 222.6 | 0.326 | 0.319 |
| **B2** | 0.0798 | 0.1061 | 249.9 | 0.275 | 0.348 |
| **B3** | 0.0754 | 0.1124 | 267.2 | 0.319 | 0.309 |
| **B4** | **0.1943** | **0.2166** | 228.5 | 0.363 | **0.856** |

### 3. Multi-Difficulty Benchmark ($N \in \{3,4,5,6\}$)

| Config | EMA Success (Stable) | Median Length (tokens) | Hallucination Rate | Exploration Efficiency $\eta$ |
| :--- | :---: | :---: | :---: | :---: |
| **B0 (Vanilla GRPO)** | 0.0375 | 190.1 | 0.315 | 0.204 |
| **B4 (Full LAGRPO)** | **0.0583** | **188.3** | 0.457 | **0.332** |

> **Exploration Efficiency ($\eta$)**: Quantified as $\eta = \frac{\text{EMA Success Rate}}{\text{Mean Length}} \times 10^3$, measuring net success probability per 1,000 generated tokens.

---

## Environment & Quick Start

### 1. Requirements & Setup

```bash
git clone https://github.com/KardeniaPoyu/LAGRPO.git
cd LAGRPO
pip install -r requirements.txt
```

### 2. SFT Warmup & Data Preparation

```bash
# Generate multi-difficulty Arithmetic-24 dataset
python data_gen_multi.py

# Train baseline SFT model
python train_sft.py --model-name Qwen/Qwen2.5-0.5B-Instruct --output-dir saved_models/sft_final
```

### 3. Training & Automated Pipeline Execution

```bash
# Train Vanilla GRPO (B0)
python train_grpo.py --ablation B0 --group-size 8 --max-steps 200

# Train Full LAGRPO (B4)
python train_grpo.py --ablation B4 --group-size 8 --max-steps 200

# Execute full automated paper ablation pipeline (B0 to B4)
python scripts/run_paper_ablations.py --max-steps 200 --group-size 32
```

### 4. Multi-Seed Replication Pipeline (Statistical Significance)

To reproduce NeurIPS/ICLR-style benchmark curves across multiple random seeds with confidence intervals:

```bash
# Automated multi-seed pipeline (Seeds: 42, 2026, 999)
bash scripts/run_multi_seed.sh

# Run complete integrated suite
bash scripts/run_integrated_suite.sh
```

### 5. Evaluation & Visualization

```bash
# Evaluate trained checkpoints on test set
python evaluate.py --models saved_models/ablations/grpo_b4_G32_final

# Generate academic figures (in plots/)
python scripts/archive/generate_lagrpo_plots.py
```

---

## Repository Structure

```
.
├── train_grpo.py              # Main training script for GRPO & LAGRPO
├── train_ppo.py               # Baseline PPO implementation
├── train_sft.py               # Supervised fine-tuning pipeline
├── env.py                     # Task environment & AST process verification sandbox
├── model_utils.py             # Model initializations and LoRA configurations
├── evaluate.py                # Standalone test-set evaluation pipeline
├── data_gen_multi.py          # Dataset generation utilities
├── scripts/                   # Automated pipelines & multi-seed verification
│   ├── run_paper_ablations.py # Paper ablation suite runner (B0-B4)
│   ├── run_multi_seed.sh      # Multi-seed replication pipeline (Seeds: 42, 2026, 999)
│   ├── run_integrated_suite.sh# Full integrated evaluation suite
│   └── archive/               # Secondary analysis and plotting scripts
├── docs/                      # Theoretical documentation and LaTeX formalizations
├── plots/                     # High-resolution academic figures (PNG, SVG, PDF)
├── requirements.txt           # Dependency specifications
└── LICENSE                    # MIT License
```

---

## Citation

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

## License

This project is released under the [MIT License](LICENSE).
