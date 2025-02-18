import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# 定义一个函数用于显示并保存图像
def save_image(image, save_path):
    image = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
    image = Image.fromarray(image.astype(np.uint8))
    image.save(save_path)

# 读取本地图片
img_path1 = "./fig1/ILSVRC2012_val_00028914.JPEG"  # 请替换为你的本地图片路径
img_path2 = "./fig1/n01440764_3347.JPEG"  # 请替换为你的本地图片路径
img_path3 = "./fig1/n03000684_2988.JPEG"  # 请替换为你的本地图片路径
img_path4 = "./fig1/n03888257_17206.JPEG"  # 请替换为你的本地图片路径
img_path5 =  "./fig1/n03000684_35647.JPEG"

img_path=img_path5


img = Image.open(img_path)
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# 归一化
img = img - imagenet_mean
img = img / imagenet_std

# 加载预训练模型
chkpt_dir = "mae_visualize_vit_large.pth"
if not os.path.exists(chkpt_dir):
    raise FileNotFoundError(f"预训练模型 {chkpt_dir} 不存在，请先下载！")

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=False)
    return model

model_mae = prepare_model(chkpt_dir, "mae_vit_large_patch16")

# 运行 MAE 处理
def run_and_save_image(img, model, save_dir="output_images"):
    if model is None:
        raise ValueError("模型加载失败，请检查 `prepare_model`")

    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

    x = torch.tensor(img).unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    save_image(x[0], os.path.join(save_dir, "original.png"))
    save_image(im_masked[0], os.path.join(save_dir, "masked.png"))
    save_image(y[0], os.path.join(save_dir, "reconstruction.png"))
    save_image(im_paste[0], os.path.join(save_dir, "reconstruction_visible.png"))

    print(f"所有结果已保存在 {save_dir}")

# 执行
run_and_save_image(img, model_mae)
