import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 确保已经安装了必要的依赖库
try:
    import requests
    import wget
except ImportError:
    print("正在安装缺失的模块...")
    os.system("pip install requests wget")

# 引入 MAE 模型
import models_mae  # 确保 'models_mae.py' 在同一目录或正确的 Python 路径中

# 检查模型文件是否存在
chkpt_dir = "mae_visualize_vit_large.pth"
if not os.path.exists(chkpt_dir):
    print(f"{chkpt_dir} 不存在，开始下载...")
    os.system(f"wget -nc https://dl.fbaipublicfiles.com/mae/visualize/{chkpt_dir}")

print(f"{chkpt_dir} 存在: {os.path.exists(chkpt_dir)}")


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    try:
        # 构建模型
        model = getattr(models_mae, arch)()

        # 加载模型权重
        checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=True)  # 新代码
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

        return model

    except AttributeError as e:
        print(f"模型构建错误: {e}")
    except FileNotFoundError as e:
        print(f"权重文件加载错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")

    return None


# 加载模型
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')

if model_mae is None:
    print("模型加载失败！")
else:
    print("模型加载成功！")

