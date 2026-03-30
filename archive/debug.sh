#!/bin/bash
#SBATCH --output=sleep_debug.out
#SBATCH --error=sleep_debug.err
#SBATCH --gres=gpu:1           # 申请 1 块 GPU
#SBATCH --time=12:00:00         # 任务持续 12 小时
#SBATCH --mem=16G              # 申请 16GB 内存
#SBATCH --cpus-per-task=4      # 申请 4 个 CPU 核心

# 让任务在后台运行 2 小时
sleep 43200
EOF

