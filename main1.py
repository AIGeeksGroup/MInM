# go for exercise

#%% md
### Prepare

import sys
import os
import requests
import wget
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import models_mae
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# define the utils
def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=True)  # 添加 weights_only 参数
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Checkpoint loaded with message: {msg}")
    return model

def run_one_image(img, model):
    if model is None:
        print("Error: Model is None. Please check the model loading process.")
        return

    try:
        x = torch.tensor(img)

        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        # run MAE
        loss, y, mask = model(x.float(), mask_ratio=0.75)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', x)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [12, 12]
        plt.rcParams['axes.titlesize'] = 48

        plt.subplot(2, 2, 1)  # 第一行第一列
        show_image(x[0], "original")

        plt.subplot(2, 2, 2)  # 第一行第二列
        show_image(im_masked[0], "masked")

        plt.subplot(2, 2, 3)  # 第二行第一列
        show_image(y[0], "reconstruction")

        plt.subplot(2, 2, 4)  # 第二行第二列
        show_image(im_paste[0], "reconstruction + visible")

        plt.show()
    except Exception as e:
        print(f"Error during inference: {e}")

# load an image
img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

# Download pretrained model checkpoint
checkpoint_url = "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth"
chkpt_dir = "mae_visualize_vit_large.pth"

if not os.path.exists(chkpt_dir):
    wget.download(checkpoint_url)

# Load pretrained model
model_mae = prepare_model(chkpt_dir, "mae_vit_large_patch16")
if model_mae:
    print("Model loaded successfully.")
else:
    print("Model loading failed.")

# Make random mask reproducible (comment out to make it change)
torch.manual_seed(2)

print('MAE with pixel reconstruction:')
run_one_image(img, model_mae)
