#!/bin/bash
#SBATCH --job-name=train              # 作业名
#SBATCH --output=output_%j.log            # 输出日志文件，%j是作业ID
#SBATCH --error=error_%j.log             # 错误日志文件，%j是作业ID
#SBATCH --ntasks=1                       # 任务数
#SBATCH --cpus-per-task=10                # 每个任务的CPU核心数
#SBATCH --gres=gpu:2                  # 请求GPU（可以根据需求调整）
#SBATCH --time=72:00:00                  # 设置作业的最大运行时间
#SBATCH --mem=64G                        # 分配内存大小

# 进入你的项目目录
cd $SLURM_SUBMIT_DIR

# 激活anaconda环境
conda init
source /home/ytia0661@acfr.usyd.edu.au/miniconda3/etc/profile.d/conda.sh
eval "$(/home/ytia0661@acfr.usyd.edu.au/miniconda3/bin/conda shell.bash hook)"

# 激活gpupytorch环境
conda init
conda activate gpupytorch

export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# 运行train1.py
torchrun --nproc_per_node=2 trainwandb5.py


