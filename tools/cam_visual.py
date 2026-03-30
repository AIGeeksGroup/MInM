import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from models import models_mae2 as models_mae
import util.misc as misc

# 1. 定义参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'mae_vit_base_patch16'  # 与训练一致
input_size = 224
checkpoint_path = './MINM_weight/best_acc.pth'  # 你的权重路径
image_path = './figm/1.JPEG'  # 你的图片路径
mask_path = './figm/1-m.JPEG'  # 你的掩码路径（可选）

# 2. 加载预训练模型
model = models_mae.__dict__[model_name]()  # 创建模型实例
model.to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])  # 加载权重
model.eval()

# 3. 图片预处理
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]


def forward_encoder_no_mask(model, x):
    x = model.patch_embed(x)  # [1, 196, 768]
    x = x + model.pos_embed[:, 1:, :]  # 加位置编码
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)  # [1, 197, 768]
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x

# 5. 获取 encoder 输出
with torch.no_grad():
    latent = forward_encoder_no_mask(model, image_tensor)  # [1, 197, 768]

# 6. 生成热力图
latent = latent[:, 1:, :]  # [1, 196, 768]，移除 CLS token
heatmap = latent.mean(dim=-1).squeeze(0)  # [196]，对特征维度取平均
heatmap = heatmap.view(14, 14)  # 重塑为 14x14
heatmap = F.interpolate(
    heatmap.unsqueeze(0).unsqueeze(0),  # [1, 1, 14, 14]
    size=(input_size, input_size),
    mode='bilinear',
    align_corners=False
).squeeze().cpu().numpy()  # [224, 224]

# 7. 可视化和保存
image_np = np.array(image.resize((input_size, input_size)))
output_dir = './heatmap'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 保存原始图片
plt.figure(figsize=(5, 5))  # 设置图片大小，可以根据需要调整
plt.imshow(image_np)
# plt.title('Original Image')
plt.axis('off')
plt.savefig('./heatmapminm/1.png', bbox_inches='tight', dpi=300)  # 高分辨率保存
plt.close()  # 关闭当前图形，避免内存占用

# 保存热力图
plt.figure(figsize=(5, 5))  # 设置图片大小，与原图一致
plt.imshow(image_np)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
# plt.title('Encoder Heatmap')
plt.axis('off')
plt.savefig('./heatmapminm/1-heatmap.png', bbox_inches='tight', dpi=300)  # 高分辨率保存
plt.close()  # 关闭当前图形

# 可选：显示图片（如果需要在运行时查看）
# plt.figure(figsize=(5, 5))
# plt.imshow(image_np)
# plt.title('Original Image')
# plt.axis('off')
# plt.show()
#
# plt.figure(figsize=(5, 5))
# plt.imshow(image_np)
# plt.imshow(heatmap, cmap='jet', alpha=0.5)
# plt.title('Encoder Heatmap')
# plt.axis('off')
# plt.show()