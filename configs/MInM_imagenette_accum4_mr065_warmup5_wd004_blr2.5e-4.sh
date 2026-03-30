#!/bin/bash
#SBATCH --job-name=minm_accum4_blr2.5e-4  # 作业名
#SBATCH --output=MInM_imagenette_accum4_mr065_warmup5_wd004_blr2.5e-4_%j.log   # 输出日志文件
#SBATCH --error=MInM_imagenette_accum4_mr065_warmup5_wd004_blr2.5e-4_%j.err   # 错误日志文件
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

# MInM tuned on imagenette — acc1 improvement experiment v2
# 相比 base (mr075_blr1e3_wd005):
#   accum_iter 1→4: eff_batch 32→128 (不增加GPU时间)
#   warmup_epochs 10→5: 更多 epoch 在 peak LR 训练
#   mask_ratio 0.75→0.65: encoder 每步看更多 patch, 小数据集短训练更有利
#   weight_decay 0.05→0.04: 小数据集稍降正则避免欠拟合
#   blr 1e-3→2.5e-4: actual_lr = 2.5e-4 * 128/256 = 1.25e-4 (与base持平, 避免LR过高)
python tools/train_imagenette.py \
    --batch_size 32 \
    --epochs 100 \
    --accum_iter 4 \
    --warmup_epochs 5 \
    --num_workers 4 \
    --save_interval 5 \
    --output_dir output_imagenette_accum4_mr065_warmup5_wd004_blr2.5e-4 \
    --log_dir output_imagenette_accum4_mr065_warmup5_wd004_blr2.5e-4 \
    --nb_classes 10 \
    --mask_ratio 0.65 \
    --blr 2.5e-4 \
    --weight_decay 0.04
