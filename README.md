# Introduction

_The link to our paper is not yet available due to pending publication._

This repository provides the core implementation of CRAFT: Cost-Aware Expert Replica Allocation with Fine-Grained
Layerwise Estimations.
The code consumes environment configurations together with an input expert-load distribution pickle file (provided in
the `traces` directory).
It performs expert-replication benefit analysis and computes a layer-wise expert-replica allocation plan that mitigates
expert load imbalance
while maintaining high memory efficiency. Please refer to our paper for methodological details.

# Setup

## Prerequisites

- **OS:** Linux or Windows 11
- **Hardware:** CPU-only
- **Software:**
    - Conda
    - Git-LFS
- **Traces:** We provide expert load distribution traces collected from DeepSeek-R1-671B (61 layers, 58 MoE layers, 256
  experts, top-8 routing) and Kimi-K2-1000B (61 layers, 60 MoE layers, 384 experts, top-8 routing) across 3000 inference
  batches in the `traces` directory (see paper for details). You may also use custom traces by setting the appropriate
  arguments.

## Installation

```
git clone https://github.com/Accelsnow/CRAFT_core.git
cd CRAFT_core
conda env create --file environment.yml
conda activate craft_core
```

If Git-LFS is not installed, `git clone` will issue a warning about missing `git-lfs`.
In this case, continue with conda environment creation and activation. After activating the conda environment, run:

```
git reset --hard
```

# Single Run

Example Configuration:

- DeepSeek-R1-671B model
- 8-node cluster (64 GPUs)
- `traces/DE.pkl` input expert load distribution

```shell
python craft_core.py --nodes 8 --experts 256 --layers 61 --first-moe-layer 3 --dist-file .\traces\DE.pkl
```

For a full list of configuration options:

```shell
python craft_core.py --help
```

# Batched Run

Windows:

```powershell
.\run_all.ps1
```

Linux:

```shell
./run_all.sh
```

# Results

The script outputs the following:

- The CRAFT replica allocation plan detailing the number of replicas allocated to each layer.
- The relative replication memory savings of CRAFT compared with EPLB (quantified by the replication ratio $R$).
- The load balancedness of placement-only EPLB, EPLB, and CRAFT.

# References

The core expert-placement and expert-replication logic in `eplb_craft.py` was developed with reference to the original
[Expert Parallelism Load Balancer (EPLB)](https://github.com/deepseek-ai/EPLB) from DeepSeek-AI (Liu et al., 2024).

Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. DeepSeek-V3
Technical Report. arXiv preprint arXiv:2412.19437, 2024.

# BibTeX for CRAFT

_WIP_
