#!/bin/bash
#SBATCH --job-name=sam_imagewoof         # 作业名
#SBATCH --output=sam_output_%j.log       # 输出日志
#SBATCH --error=sam_error_%j.log         # 错误日志
#SBATCH --ntasks=1                       # 任务数
#SBATCH --cpus-per-task=10               # CPU核心数
#SBATCH --gres=gpu:1                     # 只需1个GPU做SAM推理
#SBATCH --time=24:00:00                  # 最大运行时间
#SBATCH --mem=64G                        # 内存

cd $SLURM_SUBMIT_DIR
nvidia-smi

source /home/ywan0794/miniconda3/etc/profile.d/conda.sh
conda activate minm

# 验证 numpy 版本（torch 2.1.0 需要 numpy<2.0）
python -c "import numpy; print('numpy:', numpy.__version__); import torch; print('torch:', torch.__version__)"

# SAM checkpoint（需提前在登录节点下载到此路径）
SAM_CKPT="/home/ywan0794/MInM-code/data/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_CKPT" ]; then
    echo "ERROR: SAM checkpoint not found at $SAM_CKPT"
    echo "Please download it on the login node first:"
    echo "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ~/MInM-code/data/"
    exit 1
fi

# 完整流程：下载 imagewoof -> SAM 推理 -> 生成 sam075/masks_applied
python tools/generate_sam_masks.py \
    --data_dir data/imagewoof \
    --sam_checkpoint "$SAM_CKPT" \
    --sam_model_type vit_h \
    --device cuda
