#!/bin/bash
#SBATCH --job-name=minm_bugfix          # 作业名
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

nvidia-smi
export MASTER_ADDR="localhost"
export MASTER_PORT="29502"

# MInM bug-fix test on imagenette
# 小批量训练 + 验证: batch_size=32, epochs=10, 每5个epoch做linear probing
# 数据来源: data/imagenette/ 下的 tar.gz 文件（自动解压）
python tools/train_imagenette.py \
    --batch_size 32 \
    --epochs 800 \
    --warmup_epochs 30 \
    --num_workers 4 \
    --save_interval 50 \
    --output_dir output_imagenette_tuned \
    --log_dir output_imagenette_tuned \
    --nb_classes 10 \
    --mask_ratio 0.75 \
    --blr 5e-4 \
    --weight_decay 0.05
