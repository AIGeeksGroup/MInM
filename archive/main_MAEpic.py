
import sys
import os
import requests

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

img_path = "./fig1/ILSVRC2012_val_00028914.JPEG"  # 请替换为你的本地图片路径

import models_mae

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

def run_one_image(img, model, output_dir='output', prefix='mae'):
    x = torch.tensor(img).unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    
    loss, y, mask = model(x.float(), mask_ratio=0.75)
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

    save_image(x[0], f"{prefix}_original.png")
    save_image(im_masked[0], f"{prefix}_masked.png")
    save_image(y[0], f"{prefix}_reconstruction.png")
    save_image(im_paste[0], f"{prefix}_reconstruction_visible.png")

    print(f"Saved to folder: {output_dir}")

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
run_one_image(img, model_mae,  output_dir='output/mae_base', prefix='mae_base')
