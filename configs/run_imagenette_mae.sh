#!/bin/bash
#SBATCH --job-name=mae_baseline          # 作业名
#SBATCH --output=output_%j.log          # 输出日志文件
#SBATCH --error=error_%j.log            # 错误日志文件
#SBATCH --ntasks=1                      # 任务数
#SBATCH --cpus-per-task=10              # 每个任务的CPU核心数
#SBATCH --gres=gpu:2                    # 请求2个GPU
#SBATCH --time=72:00:00                 # 最大运行时间
#SBATCH --mem=64G                       # 分配内存
#SBATCH --nodelist=erinyes              # 指定节点

# 进入项目目录
cd $SLURM_SUBMIT_DIR
nvidia-smi

# 激活 conda 环境
source /home/ywan0794/miniconda3/etc/profile.d/conda.sh
conda activate minm

nvidia-smi
export MASTER_ADDR="localhost"
export MASTER_PORT="29503"

# MAE baseline (75% random masking) on imagenette — 对照实验
# 与 MInM 使用相同的训练参数，唯一区别是 masking 策略（随机 vs 实例引导）
python tools/train_imagenette_mae.py \
    --batch_size 32 \
    --epochs 100 \
    --warmup_epochs 10 \
    --num_workers 4 \
    --save_interval 5 \
    --mask_ratio 0.75 \
    --output_dir output_mae_imagenette \
    --log_dir output_mae_imagenette \
    --nb_classes 10
