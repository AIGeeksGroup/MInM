# 1.安装huggingface_hub
# pip install huggingface_hub
import os
from huggingface_hub import snapshot_download
 
# 使用cache_dir参数，将模型/数据集保存到指定“本地路径”
snapshot_download(repo_id="mlx-vision/imagenet-1k", repo_type="dataset",
                  cache_dir="/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/data/image_1k",
                  local_dir_use_symlinks=False, resume_download=True,
                  token='HF_TOKEN_REMOVED')
