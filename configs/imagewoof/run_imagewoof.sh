#!/bin/bash
#SBATCH --job-name=minm_imagewoof        # 作业名
#SBATCH --output=output_%j.log          # 输出日志文件
#SBATCH --error=error_%j.log            # 错误日志文件
#SBATCH --ntasks=1                      # 任务数
#SBATCH --cpus-per-task=10              # 每个任务的CPU核心数
#SBATCH --gres=gpu:2                    # 请求2个GPU
#SBATCH --time=72:00:00                 # 最大运行时间
#SBATCH --mem=64G                       # 分配内存

# 进入项目目录
cd $SLURM_SUBMIT_DIR
nvidia-smi

# 激活 conda 环境
source /home/ywan0794/miniconda3/etc/profile.d/conda.sh
conda activate minm

# 确保 HuggingFace datasets 库已安装（用于下载 imagewoof）
pip install datasets -q

nvidia-smi
export MASTER_ADDR="localhost"
export MASTER_PORT="29503"

# MInM training on imagewoof
# 数据自动从 HuggingFace frgfm/imagewoof 下载
# 小批量训练 + 验证: batch_size=32, epochs=100, 每5个epoch做linear probing
python tools/train_imagewoof.py \
    --batch_size 32 \
    --epochs 100 \
    --warmup_epochs 10 \
    --num_workers 4 \
    --save_interval 5 \
    --output_dir output_bugfix_imagewoof \
    --log_dir output_bugfix_imagewoof \
    --nb_classes 10
