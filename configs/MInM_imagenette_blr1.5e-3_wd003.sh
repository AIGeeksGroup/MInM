#!/bin/bash
#SBATCH --job-name=minm_blr1.5e-3_wd003  # 作业名
#SBATCH --output=MInM_imagenette_blr1.5e-3_wd003_%j.log   # 输出日志文件
#SBATCH --error=MInM_imagenette_blr1.5e-3_wd003_%j.err    # 错误日志文件
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

# MInM on imagenette — blr/wd tuning only
# 相比 base (blr1e-3, wd0.05, acc1=58.80%):
#   blr 1e-3→1.5e-3: actual_lr 1.25e-4→1.875e-4, 加速收敛
#   weight_decay 0.05→0.03: 小数据集减轻正则, 缓解欠拟合
#   其余参数完全不变
python tools/train_imagenette.py \
    --batch_size 32 \
    --epochs 100 \
    --warmup_epochs 10 \
    --num_workers 4 \
    --save_interval 5 \
    --output_dir output_imagenette_blr1.5e-3_wd003 \
    --log_dir output_imagenette_blr1.5e-3_wd003 \
    --nb_classes 10 \
    --mask_ratio 0.75 \
    --blr 1.5e-3 \
    --weight_decay 0.03
