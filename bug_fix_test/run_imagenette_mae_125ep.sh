#!/bin/bash
#SBATCH --job-name=mae_125ep             # 作业名
#SBATCH --output=MAE_imagenette_125ep_%j.out   # 输出日志文件
#SBATCH --error=MAE_imagenette_125ep_%j.err    # 错误日志文件
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
export MASTER_PORT="29504"

# MAE baseline: resume from epoch 100, continue to 125 epochs
python bug_fix_test/train_imagenette_mae.py \
    --batch_size 32 \
    --epochs 125 \
    --warmup_epochs 10 \
    --num_workers 4 \
    --save_interval 5 \
    --mask_ratio 0.75 \
    --resume output_mae_imagenette/checkpoint_epoch_100.pth \
    --output_dir output_mae_imagenette_125ep \
    --log_dir output_mae_imagenette_125ep \
    --nb_classes 10
