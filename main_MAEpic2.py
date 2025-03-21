
import sys
import os
import requests

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

img_path = "/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/fig1/n03000684_35647.JPEG"  # 请替换为你的本地图片路径
#/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/fig1/n01440764_3347.JPEG
#/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/fig1/n03000684_2988.JPEG
#/home/ytia0661@acfr.usyd.edu.au/PycharmProjects/MAE/mae/fig1/n03000684_35647.JPEG



import models_mae2 as models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, output_dir='output', prefix='mae', mode='bottom'):   
    if mode == 'bottom':
        x = torch.tensor(img).unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)
        custom_mask = get_bottom_half_mask().to(x.device)

    if mode == 'border':
        img = resize_and_center_image(img)
        x = torch.tensor(img).unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)
        custom_mask = get_border_mask().to(x.device)

    print("Mask shape:", custom_mask.shape)  # 预期 (1, 196)
    print("Mask values:", custom_mask.view(14, 14))  # 观察mask分布


    loss, y, mask = model(x.float(), mask_ratio=0.75, custom_mask=custom_mask)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存函数
    def save_image(tensor_image, filename):
        image_np = torch.clip((tensor_image * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy().astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        image_pil.save(os.path.join(output_dir, filename))

    save_image(x[0], f"{mode}_original.png")
    save_image(im_masked[0], f"{mode}_masked.png")
    save_image(y[0], f"{mode}_reconstruction.png")
    save_image(im_paste[0], f"{mode}_reconstruction_visible.png")

    print(f"Saved to folder: {output_dir}")

def get_bottom_half_mask(img_size=224, patch_size=16, device='cpu'):
    """
    Return a binary mask tensor of shape [1, N_patches] where:
    - 0 = keep (top half)
    - 1 = mask (bottom half)
    """
    num_patches_per_row = img_size // patch_size  # e.g., 14
    total_patches = num_patches_per_row ** 2      # e.g., 196

    mask = torch.zeros(total_patches, device=device)

    # 计算 bottom-half 的 patch 索引
    for row in range(num_patches_per_row // 2, num_patches_per_row):
        for col in range(num_patches_per_row):
            patch_index = row * num_patches_per_row + col
            mask[patch_index] = 1

    mask = mask.unsqueeze(0)  # 加上 batch 维度 [1, N_patches]
    return mask

def get_border_mask(img_size=224, patch_size=16, device='cpu'):
    """
    生成一个用于 MAE 的边框 mask：
    - 192x192 的中心区域保持不变
    - 16x16 patch 宽度的外圈 mask
    """
    num_patches = img_size // patch_size  # 14 patches in one row/column
    mask = torch.ones((num_patches, num_patches), device=device)  # 默认全 mask
    
    # **解锁中心 192×192 区域（即 1 patch 的边距）**
    mask[1:-1, 1:-1] = 0

    return mask.flatten().unsqueeze(0)  # **修正：展平成 [1, 196]**

def resize_and_center_image(img, target_size=224, inner_size=192):
    """
    将输入图像缩放到 `inner_size x inner_size`，然后放到 `target_size x target_size` 的中心，
    周围填充 16 像素的黑色区域。
    
    参数：
    - img: [H, W, 3] numpy array (预期形状为 224x224)
    - target_size: 最终输出尺寸（默认为 224）
    - inner_size: 缩放后的图像尺寸（默认为 192）

    返回：
    - centered_img: 处理后的图像 (224x224)
    """
    img_pil = Image.fromarray((img * 255).astype(np.uint8))  # 转换回 PIL 格式
    img_pil = img_pil.resize((inner_size, inner_size))  # 缩放到 192x192

    # 创建 224x224 的黑色背景
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    # 计算中心偏移量
    offset = (target_size - inner_size) // 2
    new_img.paste(img_pil, (offset, offset))  # 将 192x192 图像粘贴到 224x224 中心

    return np.array(new_img) / 255.0  # 归一化回 0-1

# Load image
img = Image.open(img_path).convert('RGB')
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

img = (img - imagenet_mean) / imagenet_std

plt.rcParams['figure.figsize'] = [5, 5]
show_image(torch.tensor(img))

# Run MAE base model
model_mae = prepare_model('mae_visualize_vit_large.pth', 'mae_vit_large_patch16')
torch.manual_seed(2)
print('MAE with pixel reconstruction:')
run_one_image(img, model_mae,  output_dir='output/mae_base', prefix='mae_base', mode='bottom')
