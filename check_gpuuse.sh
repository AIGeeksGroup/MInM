#!/bin/bash
#SBATCH --job-name=check-gpu         # 作业名
#SBATCH --output=check_gpu_%j.log    # 输出文件
#SBATCH --error=check_gpu_%j.err     # 错误输出
#SBATCH --partition=hpc              # 分区
#SBATCH --nodelist=erinyes           # 指定节点
#SBATCH --ntasks=1                   # 任务数
#SBATCH --cpus-per-task=1            # CPU核心
#SBATCH --gres=gpu:1                 # 请求一个GPU
#SBATCH --time=00:05:00              # 最多运行5分钟
#SBATCH --mem=1G                     # 内存需求

echo "===== 📍 当前节点 ====="
hostname
echo

echo "===== 📊 GPU 使用状态（nvidia-smi） ====="
nvidia-smi
echo

echo "===== 👤 当前用户相关进程（ps aux） ====="
ps aux | grep $USER | grep -v grep
echo

echo "===== 🧪 检查 CUDA 可用性（Python） ====="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
